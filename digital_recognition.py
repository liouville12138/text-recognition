import torch
import os
import gzip
import numpy as np
from torchvision import transforms
from torchsummary import summary
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, input_size: tuple, output_num):
        super(Net, self).__init__()
        self.cnn = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Conv2d(in_channels=1,
                            out_channels=8,
                            kernel_size=(3, 3),
                            stride=(1, 1),  # 卷积核移动步长
                            padding=(1, 1),  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=(1, 1),  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(8),

            torch.nn.Conv2d(in_channels=8,
                            out_channels=64,
                            kernel_size=(3, 3),
                            stride=(1, 1),  # 卷积核移动步长
                            padding=(1, 1),  # 处理边界时填充0的数量, 默认为0(不填充).
                            dilation=(1, 1),  # 采样间隔数量, 默认为1, 无间隔采样.
                            bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.BatchNorm2d(64),
        )
        full_connect_input = int(input_size[0] / 4 * input_size[1] / 4 * 64)
        self.full_connect = torch.nn.Sequential(  # (batch_size , 3, input_size)
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, full_connect_input),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(full_connect_input, output_num),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.cnn.forward(x)
        x = x.view(x.size(0), -1)  # 保留batch
        x = self.full_connect.forward(x)
        x = x.squeeze(-1)
        return x


class DigitalRecognition(object):
    def __init__(self, input_size: tuple, learning_rate: float):
        self.net = Net(input_size, 10)
        summary(self.net, input_size=(1, input_size[0], input_size[1]))
        self.loss_func = torch.nn.CrossEntropyLoss()  # 交叉熵损失不支持独热编码多位输入 自带独热编码
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

        self.statistics = {}
        for i in range(10):
            self.statistics.update({
                str(i): {
                    "name": i,
                    "test_count": 0,
                    "predict_count": 0,
                    "correct_count": 0,
                    "recall": 0.0,
                    "precision": 0.0
                }
            })

        print(self.net)

    @staticmethod
    def one_hot_decoder(data_in):
        out = []
        array_data = data_in.numpy()
        for i in range(array_data.shape[0]):
            is_appended = False
            for j in range(array_data[i].shape[0]):
                if array_data[i][j] != 0:
                    out.append(j)
                    is_appended = True
                    break
            if not is_appended:
                out.append(0)
        return torch.from_numpy(np.array(out))

    def train(self, loader, start, epochs: int):
        # 训练过程
        for epoch in range(start, epochs):
            print("train epoch : {}".format(epoch))
            avg_loss = 0
            for step, (batch_data, batch_label) in enumerate(loader):
                out = self.net(batch_data.to(torch.float32))
                prediction = F.softmax(out, dim=1)  # 行代表同一组数据不同概率，行独热编码
                loss = self.loss_func(prediction, batch_label.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
                avg_loss += loss
            avg_loss = avg_loss / len(loader)
            print("{}{} ".format("train avg loss : ", avg_loss))

            if epoch % 20 == 0:
                checkpoint = {
                    "net": self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(checkpoint, '../checkpoint/cnn/checkpoint_epoch_%s.pth' % (str(epoch)))

    def test(self, loader):
        for step, (batch_data, batch_label) in enumerate(loader):
            out = self.net(batch_data.to(torch.float32))
            test = F.softmax(out, dim=1).detach().numpy()
            predict = np.argmax(test, axis=1)
            for i in range(batch_label.shape[0]):
                print("predict: {}; origin: {};".format(predict[i], int(batch_label[i])))
                self.data_statistics(str(predict[i]), str(int(batch_label[i])), int(batch_label[i]))
        return self.statistics

    def data_statistics(self, predict, label_in, name):
        self.statistics[label_in]["name"] = name
        self.statistics[label_in]["test_count"] += 1
        self.statistics[predict]["predict_count"] += 1
        if predict == label_in:
            self.statistics[label_in]["correct_count"] += 1
        if self.statistics[label_in]["correct_count"] != 0:
            self.statistics[label_in]["precision"] = self.statistics[label_in]["correct_count"] / \
                                                     self.statistics[label_in]["test_count"]
            self.statistics[label_in]["recall"] = self.statistics[label_in]["correct_count"] / \
                                                  self.statistics[label_in]["predict_count"]

    def save_net(self, path):
        torch.save(self.net, path)

    def load_net(self, path):
        self.net = torch.load(path)

    def start_from_checkpoint(self, path):
        checkpoint_loaded = torch.load(path)  # 加载断点
        self.net.load_state_dict(checkpoint_loaded['net'])  # 加载模型可学习参数
        self.optimizer.load_state_dict(checkpoint_loaded['optimizer'])  # 加载优化器参数
        start = checkpoint_loaded['epoch']  # 设置开始的epoch
        return start


def load_data(file_path, offset):
    with gzip.open(file_path, 'rb') as file_in:
        data_out = np.frombuffer(file_in.read(), dtype=np.uint8, offset=offset)
    return data_out


if __name__ == '__main__':
    current_filepath = os.path.abspath(os.path.dirname(__file__))
    workspace = os.path.join(current_filepath, "../")

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    label_train = load_data(os.path.join(workspace, "data_set/origin/train-labels-idx1-ubyte.gz"), 8).astype(np.float32)
    data_train = load_data(os.path.join(workspace, "data_set/origin/train-images-idx3-ubyte.gz"), 16).reshape(
        len(label_train), 1, 28, 28).astype(np.float32)
    label_test = load_data(os.path.join(workspace, "data_set/origin/t10k-labels-idx1-ubyte.gz"), 8).astype(np.float32)
    data_test = load_data(os.path.join(workspace, "data_set/origin/t10k-images-idx3-ubyte.gz"), 16).reshape(
        len(label_test), 1, 28, 28).astype(np.float32)

    torch_train_set = torch.utils.data.TensorDataset(torch.from_numpy(data_train),
                                                     torch.from_numpy(label_train))
    torch_test_set = torch.utils.data.TensorDataset(torch.from_numpy(data_test),
                                                    torch.from_numpy(label_test))

    train_loader = torch.utils.data.DataLoader(
        dataset=torch_train_set,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=torch_test_set,
        batch_size=64,
        shuffle=True,
        num_workers=6,
        pin_memory=False,
    )

    image_recognition = DigitalRecognition((28, 28), 0.00001)
    # start_epoch = 0
    start_epoch = image_recognition.start_from_checkpoint("../checkpoint/cnn/checkpoint_epoch_40.pth")
    image_recognition.train(train_loader, start_epoch, 200)
    image_recognition.save_net("../checkpoint/cnn/model.pkl")
    image_recognition.load_net("../checkpoint/cnn/model.pkl")
    test_result = image_recognition.test(test_loader)
    print(test_result)
