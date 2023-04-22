import numpy as np
from ch03.ch03_test.simple_layer import SimpleLayer
from ch03.ch03_test.my_gradient import numerical_gradient as num_grad


class Simple2LayersNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 将网络参数封装到param字典中，方便管理，初始化网络，2个权重矩阵，2个偏置
        self.param = {
            'w1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'w2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

    def predict(self, x):
        """正向传播"""

        # 正向传播
        w1, b1 = self.param['w1'], self.param['b1']
        w2, b2 = self.param['w2'], self.param['b2']

        # 输入层 ---> 隐藏层
        a1 = np.dot(x, w1) + b1
        # 经过激活函数sigmoid
        z1 = SimpleLayer.sigmoid(a1)

        # 隐藏层 ---> 输出层
        a2 = np.dot(z1, w2) + b2
        y = SimpleLayer.softmax(a2)

        return y

    def loss(self, x, t):
        """损失函数，这里使用交叉熵"""
        y = self.predict(x)
        return SimpleLayer.cross_entropy_error(y, t)

    def numerical_gradient(self, x, t):
        """数值微分求权值与偏置的梯度，可以利用之前写的现成的方法（在my_gradient.py）中"""
        grads = {}
        func = lambda W: self.loss(x, t)
        grads['w1'] = num_grad(func, self.param['w1'])
        grads['b1'] = num_grad(func, self.param['b1'])
        grads['w2'] = num_grad(func, self.param['w2'])
        grads['b2'] = num_grad(func, self.param['b2'])

        return grads

    def accuracy(self, x, t):
        """评价错误率，即预测结果和标签标识的结果相同的个数"""

        # 错误率就是预测的结果（多分类中就是最大的输出）和真实的结果对的上的个数
        # 计算精度也要进行向前传播，因为每次epoch都会使用梯度下降法进行网络参数
        # 更新，经过一定次数的epoch后使用这个函数经过网络计算一下，再将预测值与
        # 标签比较，就可以得出在数据集上（训练集和测试集）的准确率
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(np.sum(x.shape[0]))


if __name__ == '__main__':
    net = Simple2LayersNet(784, 100, 10)
    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)
    grads = net.numerical_gradient(x, t)
    print(grads)
