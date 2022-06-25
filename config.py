# -*- coding: utf-8 -*-
# @Time: 2022/6/23 20:19
# @Author: QinWei
# @FileName: config.py
# @Software: PyCharm
# @Desc: 配置文件

class Configuration:
    """
    configuration parameters for this project
    """
    in_channels = 1
    out_channels = 100
    num_classes = 2
    batch_size = 32
    epochs = 20
    log_step_freq = 50
    workers = 2
    optimizer = 'SGD'
    data_path = ''
    use_gpu = False
    result_file = '/result/result.csv'
    learning_rate = 0.01
    learning_decay = 0.95
    weight_decay = 1e-4
    time_format = "%m_%d_%H:%M:%S"
    model_file_name = ''
    model_path = ''




