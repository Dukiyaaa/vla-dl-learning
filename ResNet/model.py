import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        # 一共有两类残差块
        # 第一类：两个3x3卷积，步幅为1，不改变分辨率
        # 第二类：两个3x3卷积，步幅为2，改变分辨率，所以要加一个1x1卷积修正
        self.ReLU = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.BN1 = nn.BatchNorm2d(num_features=out_channels)
        # 第二个3x3不改变通道和分辨率
        self.c2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.BN2 = nn.BatchNorm2d(num_features=out_channels)
        if stride != 1 or in_channels != out_channels:  # 根据ai建议，加宽了判断条件
            # 1x1卷积根据第一个3x3的步幅来决定
            self.c3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)
        else:
            self.c3 = None

    def forward(self, x):
        y = x
        x = self.ReLU(self.BN1(self.c1(x)))
        x = self.BN2(self.c2(x))  # 这里不接relu，是因为残差相加后会再进行一次relu
        if self.c3 is not None:
            y = self.c3(y)

        x = self.ReLU(x + y)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 仿照VGG进行分块构建
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )

        self.b2 = nn.Sequential(
            Residual(in_channels=64, out_channels=64),
        )

        self.b3 = nn.Sequential(
            Residual(in_channels=64, out_channels=64),
        )

        self.b4 = nn.Sequential(
            Residual(in_channels=64, out_channels=128, stride=2),
            Residual(in_channels=128, out_channels=128),
        )

        self.b5 = nn.Sequential(
            Residual(in_channels=128, out_channels=256, stride=2),
            Residual(in_channels=256, out_channels=256),
        )

        self.b6 = nn.Sequential(
            Residual(in_channels=256, out_channels=512, stride=2),
            Residual(in_channels=512, out_channels=512),
        )

        self.b7 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet().to(device)
    print(summary(model, input_size=(1, 224, 224)))
