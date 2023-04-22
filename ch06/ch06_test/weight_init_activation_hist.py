# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

# 初始化网络参数
input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

# 使用循环进行激活值的传递
for i in range(hidden_layer_size):
    # 记录激活值，也就是第一层地输出
    if i != 0:
        # 把前一个激活值传递给x，作为后一层的输出。
        x = activations[i-1]

    # 从输入到第一层隐藏层,直接计算激活值
    # 先初始化激活值
    # w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    # 正向传播
    a = np.dot(x, w)
    # 计算激活值
    # 这里可以改变激活函数的类型，可以探究相同输入，不同激活函数的激活值
    # z = sigmoid(a)
    z = ReLU(a)
    # z = tanh(a)

    # 保存激活值到字典中
    activations[i] = z

    # 回绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    if i != 0: plt.yticks([], [])
    plt.xlim(0, 1)
    plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0, 1))
plt.show()

