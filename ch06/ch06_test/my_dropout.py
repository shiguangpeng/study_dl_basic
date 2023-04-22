import numpy as np


class MyDropout:
    """Dropout层的实现，随机生成与目标层形状相等的矩阵，再通过矩阵中的元素与dropout_ratio比较生成删除的掩膜

        :param dropout_rate: 丢弃率

        Notes: 这里使用的是哈达玛积，对应元素相乘即可，另外在计算时mask中的False代表0，True代表1，

    """

    def __init__(self, dropout_rate=0.5):
        self.dropout_ratio = dropout_rate
        # 掩膜，用来标识要删除的节点
        self.mask = None

    def forward(self, x, train_flag):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1 - self.dropout_ratio)

    def backward(self, dout):
        np.random.uniform()
        return dout * self.mask
