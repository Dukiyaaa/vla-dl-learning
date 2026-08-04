from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',                        # 保存路径
                          train=True,
                          transform=transforms.Compose(
                              [transforms.Resize(size=224),     # 第1步：将图片放大到 224x224 像素
                               transforms.ToTensor()]),         # 第2步：将PIL图片转为PyTorch张量
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,                    # 是否打乱
                               num_workers=0)                   # 子进程数量

# 只取第一个批次的图片 step为0时便为第一批图片数据及其标签
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

# 第一批数据有64张图片及其对应的64个标签
# b_x 的形状是 (64, 1, 224, 224)，表示64张图片，每张图片为224x224x1
# b_y 的形状是 (64, ),表示64个标签
# squeeze() 的作用： 移除所有大小为1的维度，所以b_x经此处理后变为(64,224,224)
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
print(class_label)

# 可视化第一个batch的图片组
plt.figure(figsize=(12, 5))     # 创建画布：12x5英寸
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    # 创建一个 4行 × 16列 的子图网格
    # ii + 1 表示当前绘制的是第几个子图（从1开始编号）
    # 因为 4 × 16 = 64，刚好放64张图
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    # batch_x[ii, :, :] 取第 ii 张图片（224x224像素）
    # cmap=plt.cm.gray 用灰度色彩映射显示
    plt.title(class_label[batch_y[ii]], size=10)
    # batch_y本身为numpy数字，即用数字来表示label，class_label[batch_y[ii]]是一个字符串
    plt.axis("off")
    # 关闭坐标轴
    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(wspace=0.05)  # 子图之间的水平间距为0.05
plt.show()