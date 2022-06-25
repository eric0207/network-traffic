# -*- coding: utf-8 -*-
# @Time: 2022/6/23 20:06
# @Author: QinWei
# @FileName: model.py
# @Software: PyCharm
# @Desc: 网络模型

import time

import torch
from torch import nn
from torchsummary import summary


from config import Configuration

cf = Configuration()


class BaseModel(nn.Module):
    """
    Usage: 所有childs模型都应该继承这个模块，直接使用model.save()或者model.load(path)
    """

    def __init__(self):
        super(BaseModel, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        """
        以“model_name+time”的格式保存模型
        :param name:
        :return:
        """
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + cf.time_format + '.pth')
        torch.save(self.state_dict(), name)
        return name


class CnnModel(BaseModel):
    """

    """

    def __init__(self, in_channels, out_channels, classes):
        super(CnnModel, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.pre_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5,)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, classes),
            nn.ReLU(),
            nn.Softmax(classes)
        )

    def forward(self, x):
        x = self.pre_layers(x)
        shape_of_x = x.shape
        fc1 = nn.Linear(shape_of_x[-1], 16)
        fc2 = nn.Linear(16, 8)
        fc3 = nn.Softmax(self.classes)
        x = nn.ReLU(fc1(x))
        x = nn.ReLU(fc2(x))
        return fc3(x)


if __name__ == '__main__':
    test_model = CnnModel(1, 2, 4)
    summary(test_model, input_size=(1024, 1), batch_size=20, device='cpu')



