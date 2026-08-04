import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(self, in_channel, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()
        # 构造inception, 各自有四条路径;多个inception之间的区别只有通道数
        # 第一条路径：1x1卷积块
        self.p1 = nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=1)
        # 第二条路径：1x1卷积 + 3x3卷积
        self.p2_1 = nn.Conv2d(in_channels=in_channel, out_channels=c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1)
        # 第三条路径：1x1卷积 + 5x5卷积
        self.p3_1 = nn.Conv2d(in_channels=in_channel, out_channels=c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2)
        # 第四条路径：3x3最大池化 + 1x1卷积
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)    # 池化的默认stride=kernel_size
        self.p4_2 = nn.Conv2d(in_channels=in_channel, out_channels=c4, kernel_size=1)

    def forward(self, x):
        # 得到四个通道的输出，最后cat，将通道数相加
        p1 = self.ReLU(self.p1(x))  # 通道1的输出
        p2 = self.ReLU(self.p2_2(self.ReLU(self.p2_1(x))))  # 通道2的输出
        p3 = self.ReLU(self.p3_2(self.ReLU(self.p3_1(x))))  # 通道3的输出
        p4 = self.ReLU(self.p4_2(self.p4_1(x)))             # 通道4的输出,maxpool不接relu

        return torch.cat((p1, p2, p3, p4), dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        # 组建完整GoogLeNet网络，按块区分，完整网络结构得看论文
        # 第一个块：7x7卷积 + 最大池化, 输入224x224x1, 输出56x56x64
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第二个块：1x1卷积 + 3x3卷积 + 最大池化, 输入56x56x64, 输出28x28x192
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第三个块：2个inception
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第四个块：5个inception
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第五个块：2个inception
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # 第六个块：GAP(全局平均池化) + FC
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),   # 此处声明最后的输出尺寸 WxH
            nn.Flatten(),
            nn.Linear(1024, 10)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GoogLeNet()
    model = model.to(device)
    print(summary(model,(1, 224, 224)))

