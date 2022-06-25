## 深度学习网络流量CNN模型

~~~
文件结构
|-data---
|	  |--__init__.py
|	  |--load_data.py		数据加载处理
|-models---
|	    |--__init__.py
|	    |--model.py		网络模型
|-utils---
|	   |--__init__.py
|-congif.py 				配置文件
|-main.py					主文件
|-README.md
|-requirement.txt
~~~

### 网络流量识别的数据管道



![image-20220625094257531](https://github.com/eric0207/network-traffic/blob/main/images/pipeline.png)

### 一维CNN模型的架构



![image-20220625095041497](https://github.com/eric0207/network-traffic/blob/main/images/net.png)