# -*- coding: utf-8 -*-
# @Time: 2022/6/23 19:56
# @Author: QinWei
# @FileName: model.py
# @Software: PyCharm
# @Desc: 主文件

import datetime

import pandas as pd
import torch
from torch import nn, optim
from data.load_data import load_dataset, create_train_test, MyDataset
from models.model import CnnModel
from config import Configuration
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt

cf = Configuration()


def train_step(model, features, labels):
    """
    训练步骤
    :param model:
    :param features:
    :param labels:
    :return:
    """
    # 训练模式，dropout层发生作用
    model.train()

    # 梯度清零
    model.optimizer.zero_grad()

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions, labels)
    metric = model.metric_func(predictions, labels)

    # 反向传播求梯度
    loss.backward()
    model.optimizer.step()

    return loss.item(), metric.item()


def valid_step(model, features, labels):
    """
    验证步骤
    :param model:
    :param features:
    :param labels:
    :return:
    """
    # 预测模式，dropout层不发生作用
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        predictions = model(features)
        loss = model.loss_func(predictions, labels)
        metric = model.metric_func(predictions, labels)

    return loss.item(), metric.item()


def train_model(model, epochs, dl_train, dl_valid, log_step_freq):
    """
    训练模型
    :param model:
    :param epochs:
    :param dl_train:
    :param dl_valid:
    :param log_step_freq:
    :return:
    """
    metric_name = model.metric_name
    df_history = pd.DataFrame(columns=["epoch", "loss", metric_name, "val_loss", "val_" + metric_name])
    print("Start Training...")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % now_time)

    for epoch in tqdm(range(1, epochs + 1)):

        # 1，训练循环-------------------------------------------------
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for step, (features, labels) in enumerate(dl_train, 1):

            loss, metric = train_step(model, features, labels)

            # 打印batch级别日志
            loss_sum += loss
            metric_sum += metric
            if step % log_step_freq == 0:
                print(("[step = %d] loss: %.3f, " + metric_name + ": %.3f") %
                      (step, loss_sum / step, metric_sum / step))

        # 2，验证循环-------------------------------------------------
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(dl_valid, 1):
            val_loss, val_metric = valid_step(model, features, labels)

            val_loss_sum += val_loss
            val_metric_sum += val_metric

        # 3，记录日志-------------------------------------------------
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        df_history.loc[epoch - 1] = info

        # 打印epoch级别日志
        print(("\nEPOCH = %d, loss = %.3f," + metric_name + "  = %.3f, val_loss = %.3f, " +
               "val_" + metric_name + " = %.3f") % info)
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "==========" * 8 + "%s" % now_time)

    print('Finished Training...')

    return df_history


def plot_metric(df_history, metric):
    """
    作图
    :param df_history:
    :param metric:
    :return:
    """
    train_metrics = df_history[metric]
    val_metrics = df_history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


def predict(model, dl):
    """
    预测
    :param model:
    :param dl:
    :return:
    """
    model.eval()
    with torch.no_grad():
        result = torch.cat([model.forward(t[0]) for t in dl])
    return result.data


def main(optimizer):
    # 加载数据
    x, y = load_dataset(cf.data_path)
    # 创建训练测试集
    x_train, x_test, y_train, y_test = create_train_test(x, y)
    #
    x_train, x_test, y_train, y_test, class_labels, nb_classes = transform_data(x_train, x_test, y_train, y_test)

    # 构建dataLoader
    train_dataset = MyDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
    train_loader = data.DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=cf.workers)

    test_dataset = MyDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())
    test_loader = data.DataLoader(test_dataset, batch_size=cf.batch_size, shuffle=True, num_workers=cf.workers)

    # 创建模型
    model = CnnModel(cf.in_channels, cf.out_channels, nb_classes)

    if optimizer == 'SGD':
        model.optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer == 'Adam':
        model.optimizer = optim.Adam(model.parameters(), lr=0.01)
    elif optimizer == 'RMSprop':
        model.optimizer = optim.RMSprop(model.parameters(), lr=0.01, )
    elif optimizer == 'SGD-Momentum':
        model.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    elif optimizer == 'SGD-Nesterov':
        model.optimizer = optim.SGD(model.parameters(), lr=0.01, nesterov=True)

    model.loss_func = nn.CrossEntropyLoss()
    model.metric_func = lambda y_pred, y_true: roc_auc_score(y_true.data.numpy(), y_pred.data.numpy())
    model.metric_name = "accuracy"

    # 训练模型
    df_history = train_model(model, cf.epochs, train_loader, test_loader, cf.log_step_freq)

    # 评估模型
    plot_metric(df_history, "loss")
    plot_metric(dfhistory, "auc")

    # 预测模型
    # 预测概率
    y_pred_probs = predict(model, dl_valid)
    # 预测类别
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))

    # 模型保存
    model.save(cf.model_file_name)

    # 模型加载
    model.load(cf.model_path)


if __name__ == '__main__':
    main(cf.optimizer)
