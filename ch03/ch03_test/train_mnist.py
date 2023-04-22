import math

import numpy as np
import pickle as pkl
# 导入自己封装的层类
from simple_layer import SimpleLayer

import matplotlib.pyplot as plt


def draw():
    fig, ax = plt.subplots()
    x = np.array([0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.log(x)
    ax.plot(x, y)
    plt.show()


def mse(y, t):
    """
    Parameters
    ----------
    :y 网络的预测值
    :t 真实值

    :Returns 均方误差
    -------
    """
    return np.sum(0.5 * (y - t) ** 2)


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))


if __name__ == '__main__':
    # 三层神经网络（不含输出）
    # 输入
    x = np.random.randn(2, 784)
    # 隐藏层：hidden_layer50、hidden_layer100
    input_layer = SimpleLayer(x, 50, 'input')
    out1 = input_layer.forward()
    input_layer.trace_layer_params()

    hidden_layer50 = SimpleLayer(out1, 100, name='hidden50')
    out2 = hidden_layer50.forward()

    hidden_layer100 = SimpleLayer(out2, 10, name='hidden100')
    out3 = hidden_layer100.forward()

    # 输出层
    out_layer = SimpleLayer(out3, 10, activation='softmax', name='output')
    result = out_layer.forward()
