import numpy as np
from ch05.ch05_test.simple_2lyrs_net2 import Simple2LayersNet2


class SGDAlgorithm:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """

        Parameters
        ----------
        :param params: 待优化的变量字典
        :param grads: 对应params变量的梯度

        Returns params: 优化结果
        -------

        """

        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]

        return params


class MomentumAlgorithm:
    def __init__(self, momentum=0.9, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.momentum = momentum
        # 速度，需要为每一个要优化的参数增加对应的速度
        self.velocity = None

    def update(self, params, grads):
        """

        Parameters
        ----------
        :param params: 待优化的变量字典
        :param grads: 对应params变量的梯度

        Returns params: 优化结果
        -------

        """

        # velocity初始化
        if self.velocity is None:
            self.velocity = {}
            for key, value in params.items():
                self.velocity[key] = np.zeros_like(value)

        for key in params.keys():
            self.velocity = self.momentum * self.velocity - self.learning_rate * grads[key]
            params[key] += self.velocity
        return params


class AdaGradAlgorithm:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        # 梯度平方和
        self.h = None

    def update(self, params, grads):
        """

        Parameters
        ----------
        :param params: 待优化的变量字典
        :param grads: 对应params变量的梯度

        Returns params: 优化结果
        -------

        """

        # h初始化
        if self.h is None:
            self.h = {}
            for key, value in params.items():
                self.h[key] = np.zeros_like(value)

        for key in params.keys():
            self.h += grads[key] * grads[key]
            params[key] -= (self.learning_rate * grads[key]) / np.sqrt(self.h[key] + 1e-7)
        return params
