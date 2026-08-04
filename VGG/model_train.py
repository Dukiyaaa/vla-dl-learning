import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import VGG
import torch.nn as nn
import pandas as pd

def train_val_data_process():
    train_data = FashionMNIST(root='./data',  # 保存路径
                              train=True,
                              transform=transforms.Compose(
                                  [transforms.Resize(size=224),  # 第1步：将图片放大到 224x224 像素
                                   transforms.ToTensor()]),  # 第2步：将PIL图片转为PyTorch张量
                              download=True)

    train_data, val_data = Data.random_split(train_data, [round(0.8 * len(train_data)), round(0.2 * len(train_data))])

    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=32,
                                       shuffle=True,  # 是否打乱
                                       num_workers=0)  # 子进程数量

    val_dataloader = Data.DataLoader(dataset=val_data,
                                     batch_size=32,
                                     shuffle=False,  # 是否打乱
                                     num_workers=0)  # 子进程数量

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # 训练平台
    # 优化后的设备选择
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")
    # 优化器，梯度下降的优化版
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    # 先保存模型最开始的权重参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 数据记录
    best_acc = 0.0
    # 损失值记录
    train_loss_all = []
    val_loss_all = []
    # 准确度记录
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # 训练集的损失及准确度
        train_loss = 0.0
        train_corrects = 0
        # 验证集的损失及准确度
        val_loss = 0.0
        val_corrects = 0
        # 样本数量
        train_num = 0
        val_num = 0

        # 每个epoch里面都对全部的数据做统一处理
        for step, (b_x, b_y) in enumerate(train_dataloader):
            # 将数据放到设备上进行计算，与模型一致
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 启动训练模式
            model.train()

            # 推理结果，取交叉熵做损失函数 argmax 是 "argument of the maximum" 的缩写，意思是"最大值所在的位置"。
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            # 清除梯度 以下三行为通用写法
            optimizer.zero_grad()
            # 反向传播，更新参数
            loss.backward()
            optimizer.step()

            # b_x.size() 返回 torch.Size([128, 1, 224, 224])
            # loss.item()是该批次(128)个数据的平均loss！！，所以要乘size
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)

            train_num += b_x.size(0)

        # 验证集上做同样操作，也就是每一轮里，先通过训练集优化参数，再在验证集上看看这一轮优化的结果如何
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                # 将数据放到设备上进行计算，与模型一致
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                # 启动评估模式
                model.eval()

                # 推理结果，取交叉熵做损失函数 argmax 是 "argument of the maximum" 的缩写，意思是"最大值所在的位置"。
                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)

                # 直接统计结果
                # b_x.size() 返回 torch.Size([128, 1, 224, 224])
                # loss.item()是该批次(128)个数据的平均loss！！，所以要乘size
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)

                val_num += b_x.size(0)

        # 统计这一轮的各种结果
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.float().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.float().item() / val_num)

        print("{} train loss: {:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss: {:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 保存最高准确度权重参数
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    time_use = time.time() - since
    print("训练及验证花费时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, 'best_model/best_model.pth')

    # 保存panda数据帧
    train_process = pd.DataFrame(data={
        "epoch": range(num_epochs),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all,
    }
    )

    return train_process


def matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    # 先画第一个子图
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, "ro-", label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, "bs-", label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")
    # 第二个子图
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, "ro-", label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, "bs-", label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.show()


if __name__ == "__main__":
    model = VGG()
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, 20)
    matplot_acc_loss(train_process)
