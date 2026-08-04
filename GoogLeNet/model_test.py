import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import GoogLeNet
import torch.nn as nn
import pandas as pd


def test_data_process():
    test_data = FashionMNIST(root='./data',  # 保存路径
                             train=False,
                             transform=transforms.Compose(
                                 [transforms.Resize(size=224),  # 第1步：将图片放大到 28x28 像素
                                  transforms.ToTensor()]),  # 第2步：将PIL图片转为PyTorch张量
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,  # 是否打乱
                                      num_workers=0)  # 子进程数量

    return test_dataloader


def test_model_process(model, test_dataloader):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("使用 CPU")

    # model放到mps上，注意后面数据也要做同样处理
    model = model.to(device)

    # 测试参数
    test_corrects = 0.0
    test_num = 0

    # 标准化写法 由于只有前向传播，所以把梯度置0
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 评估模式
            model.eval()

            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

        # 测试准确率
        test_acc = test_corrects.float().item() / test_num
        print("测试准确率:{}".format(test_acc))


if __name__ == "__main__":
    model = GoogLeNet()
    model.load_state_dict(torch.load('best_model/best_model.pth'))
    test_dataloader = test_data_process()
    test_model_process(model, test_dataloader)

    # # 以下为推理过程可视化，目的是格式化推理结果与真实结果的对比
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print(f"使用 NVIDIA GPU: {torch.cuda.get_device_name(0)}")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("使用 Apple Silicon GPU (MPS)")
    # else:
    #     device = torch.device("cpu")
    #     print("使用 CPU")
    #
    # # model放到mps上，注意后面数据也要做同样处理
    # model = model.to(device)
    # with torch.no_grad():
    #     for test_data_x, test_data_y in test_dataloader:
    #         test_data_x = test_data_x.to(device)
    #         test_data_y = test_data_y.to(device)
    #
    #         model.eval()
    #
    #         output = model(test_data_x)
    #         pre_lab = torch.argmax(output, dim=1)
    #         result = pre_lab.item()
    #         label = test_data_y.item()
    #         print("预测值 = {}, 真实值 = {}".format(result, label))