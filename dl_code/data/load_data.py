# -*- coding: utf-8 -*-
# @Time: 2022/6/23 20:17
# @Author: QinWei
# @FileName: load_data.py
# @Software: PyCharm
# @Desc: 数据加载处理

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils import data


class MyDataset(data.Dataset):

    def __init__(self, x_train, y_train):
        super(MyDataset, self).__init__()
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

    def __len__(self):
        return len(self.x_train)


def load_dataset(data_file):
    if data_file:
        df = pd.read_csv(data_file, header=None)
        data_array = df.values
        x, y = [], []
        for data in data_array:
            x.append(data[1:])
            y.append(data[0])
        x = np.array(x)
        y = np.array(y)
    else:
        raise IOError('Non-empty filename expected.')
    return x, y


def create_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    return x_train, x_test, y_train, y_test


def transform_data(x_train, x_test, y_train, y_test):
    # reshape attributes to spatial dimensions
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    # one-hot-encode output labels (protocol names)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    y_train = pd.get_dummies(encoded_y_train)

    encoder.fit(y_test)
    class_labels = encoder.classes_  # the name of the class labels encoded
    nb_classes = len(class_labels)  # the number of different labels being trained
    encoded_y_test = encoder.transform(y_test)
    y_test = pd.get_dummies(encoded_y_test)

    return x_train, x_test, y_train, y_test, class_labels, nb_classes



