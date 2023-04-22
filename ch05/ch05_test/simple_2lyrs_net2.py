import numpy as np
from ch03.ch03_test.simple_layer import SimpleLayer
from ch03.ch03_test.my_gradient import numerical_gradient as num_grad
import MyBasicLayer as mblyr


class Simple2LayersNet2:
    """本类与Simple2LayersNet的区别是：
        引入了层的概念，将传播的节点使用`层类`来封装
        并且Simple2LayersNet中使用的是numerical_gradient数值微分的方法求梯度
        ，而本类使用反向传播求梯度

    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 将网络参数封装到param字典中，方便管理，初始化网络，2个权重矩阵，2个偏置
        self.param = {
            'w1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'w2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }

        # 引入了层的概念，因此在初始化时需要将层交给类的实例管理
        # 注意：层是有顺序的
        self.hidden_layers_dict = {'Affine_1': mblyr.AffineLayer(self.param['w1'], self.param['b1']),
                                   'Relu_1': mblyr.ReluLayer(),
                                   'Affine_2': mblyr.AffineLayer(self.param['w2'], self.param['b2'])}
        # 输出层
        self.out_layer = mblyr.SoftmaxLossLayer()

    # def predict(self, x):
    #     """正向传播"""
    #
    #     # 正向传播
    #     w1, b1 = self.param['w1'], self.param['b1']
    #     w2, b2 = self.param['w2'], self.param['b2']
    #
    #     # 输入层 ---> 隐藏层
    #     a1 = np.dot(x, w1) + b1
    #     # 经过激活函数sigmoid
    #     z1 = SimpleLayer.sigmoid(a1)
    #
    #     # 隐藏层 ---> 输出层
    #     a2 = np.dot(z1, w2) + b2
    #     y = SimpleLayer.softmax(a2)
    #
    #     return y

    def predict(self, x):
        """ 正向传播version2

        Parameters
        ----------
        x

        Returns
        -------

        """
        # # 正向传播
        # w1, b1 = self.param['w1'], self.param['b1']
        # w2, b2 = self.param['w2'], self.param['b2']
        #
        # # 输入层 ---> 隐藏层
        # a1 = np.dot(x, w1) + b1
        # # 经过激活函数sigmoid
        # z1 = SimpleLayer.sigmoid(a1)
        #
        # # 隐藏层 ---> 输出层
        # a2 = np.dot(z1, w2) + b2
        # y = SimpleLayer.softmax(a2)
        #
        # return y

        # 使用x，根据layers中的层进行正向传播，最后一层softmax-loss由于涉及到loss因此安排在loss函数中

        for layer_name, layer in self.hidden_layers_dict.items():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """损失函数，这里使用交叉熵"""
        y = self.predict(x)
        # return SimpleLayer.cross_entropy_error(y, t)
        # ver2: 这里使用已经封装的softmax-loss层
        loss = self.out_layer.forward(y, t)
        return loss

    def numerical_gradient(self, x, t):
        """数值微分求权值与偏置的梯度，可以利用之前写的现成的方法（在my_gradient.py）中"""
        grads = {}
        func = lambda W: self.loss(x, t)
        grads['w1'] = num_grad(func, self.param['w1'])
        grads['b1'] = num_grad(func, self.param['b1'])
        grads['w2'] = num_grad(func, self.param['w2'])
        grads['b2'] = num_grad(func, self.param['b2'])

        return grads

    def gradient(self, x, t):
        """反向传播的起点是损失函数对下游节点求偏导，因此需要得到预测的值y和真实标签t求出loss

        Parameters
        ----------
        y
        t

        Returns
        -------

        """

        # ======向前传播======
        self.loss(x, t)
        # ======向前传播======

        # ======向后传播 backpropagation======
        dout = 1
        # 从最后一层开始反向传播
        dout = self.out_layer.backward(dout)
        # 反向传播需要从最后一层开始计算，需要把这个字典倒转
        hidden_keys = self.hidden_layers_dict.keys()
        # 反转层的键，方便反向传播
        reversed_key = reversed(list(hidden_keys))
        for key in reversed_key:
            dout = self.hidden_layers_dict.get(key).backward(dout)
        # ======向后传播 backpropagation======

        # 保存网络参数梯度
        grads = {'w1': self.hidden_layers_dict['Affine_1'].dW, 'b1': self.hidden_layers_dict['Affine_1'].db,
                 'w2': self.hidden_layers_dict['Affine_2'].dW, 'b2': self.hidden_layers_dict['Affine_2'].db}

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
    net = Simple2LayersNet2(2, 3, 2)
    # x = np.random.rand(100, 784)
    # t = np.random.rand(100, 10)
    # grads = net.numerical_gradient(x, t)
    # print(grads)
    x = net.predict(np.array([[1., 2.], [2., 5.]]))
    print(x)
