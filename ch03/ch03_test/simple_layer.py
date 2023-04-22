import random

import numpy as np


class SimpleLayer:
    """简单层的类实现

    Warnings:
        注意：请传入2维矩阵，即使是一维向量时也要以二维矩阵的方式传入
    """

    def __init__(self, x_matrix, neural_num, activation='sigmoid', name=None):
        self.name = name
        if self.name is None:
            self.name = 'layer' + str(random.randint(1, 9999))
        self.x_matrix = x_matrix
        # 初始化层参数（权重weight 与 偏置bias）
        x_shape = x_matrix.shape[1]
        self.weight = np.random.randn(x_shape, neural_num)

        self.bias = np.zeros(neural_num)
        # 使用何种激活函数
        self.activation = activation

    def forward(self):
        # 计算总和
        a = np.dot(self.x_matrix, self.weight) + self.bias
        # 选择不同的激活函数，经过激活函数处理作为该层的输出
        if self.activation == 'relu':
            return self.relu(a)
        elif self.activation == 'softmax':
            return self.softmax(a)
        else:
            return self.sigmoid(a)

    @staticmethod
    def relu(a):
        return np.maximum(0, a)

    @staticmethod
    def sigmoid(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def _mse(y, t):
        """
        Parameters
        ----------
        :y 网络的预测值
        :t 真实值

        :Returns 均方误差
        -------
        """
        return np.sum(0.5 * (y - t) ** 2)

    @staticmethod
    def cross_entropy_error(y, t):
        # 判断是不是只有一个元素，二维表示法，但只有一个元素
        if y.ndim == 1:
            # 去掉外层的中括号，相当于将数据从二维数组中取出
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        # 若监督/标签数据是one-hot编码，需要转成真实的标签值，以便使用数值索引去找对应的概率值，这个等于就是判断t标签是不是被转化成了one-hot编码去了
        if y.size == t.size:
            t = np.argmax(t, axis=1)

        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def loss(self, y, t, name='cross_entropy_error'):
        if name == 'mse':
            return self._mse(y, t)
        else:
            return self.cross_entropy_error(y, t)

    def gradient_loss(self, x, t, name='cross_entropy_error'):
        """按照定义计算损失函数对权重矩阵的导数时，因为涉及到权重矩阵中的每个元素的改变，因此必须反复进行正向传播，这样计算出的损失对w的导数才是正确"""

        # 正向传播
        z = np.dot(x, self.weight)

        # 经过激活函数
        y = self.softmax(z)
        if name == 'mse':
            return self._mse(y, t)
        else:
            return self.cross_entropy_error(y, t)

    def trace_layer_params(self):
        print('\033[31m******* The layer {0} info are as follows:***********\033[0m'.format(self.name))
        print('===the first of weight is:\n {0}\n===bias is:\n {1}'.format(self.weight, self.bias))
        print('===the first of input variable x is: {0}'.format(self.x_matrix))
        print('=== the weight of shape is: {0}\n=== bias\' shape is: {1}'.format(self.weight.shape, self.bias.shape))
        print('=== the input variable x of shape is: {0}'.format(self.x_matrix.shape))
        print('\033[31m+++++++++++++++++++ layer end +++++++++++++++++++++++++++++\033[0m\n')

    # softmax激活函数
    @staticmethod
    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)  # 溢出对策
        return np.exp(x) / np.sum(np.exp(x))


if __name__ == '__main__':
    # # 新建层对象
    # lyr = SimpleLayer(np.array([0.6, 0.9]), 2)
    # w1 = lyr.forward()
    #
    # lyr2 = SimpleLayer(w1, 3)
    # w2 = lyr2.forward()
    # lyr2.trace_layer_params()

    # lyr3 = SimpleLayer(w3, 2)
    # val3 = lyr3.forward()
    # lyr3.trace_layer_params()
    #
    # lyr4 = SimpleLayer(val3, 2)
    # val4 = lyr4.forward()
    # lyr4.trace_layer_params()
    # print(val4)
    pass
