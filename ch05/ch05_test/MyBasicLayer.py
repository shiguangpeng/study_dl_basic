import numpy as np

from ch03.ch03_test.simple_layer import SimpleLayer


class MyMulLayer:
    """乘法层的实现，包含正向传播与反向传播

    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, d_out):
        dx = self.y * d_out
        dy = self.x * d_out
        return dx, dy


class MyAddLayer:
    """加法层的实现，包含正向传播与反向传播

    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x + y

    def backward(self, d_out):
        dx = d_out * 1
        dy = d_out * 1
        return dx, dy


class SigmoidLayer:
    """sigmoid层的正向与反向传播，relu函数只有一个自变量，即x

    """

    def __init__(self):
        self.out = None

    def forward(self, x):
        output = 1 / (1 + np.exp(-x))
        self.out = output
        return output

    def backward(self, d_out):
        d_x = d_out * self.out * (1 - self.out)
        return d_x


class ReluLayer:
    """Relu层的正向与反向传播，relu函数只有一个自变量，即x

    """

    def __init__(self):
        self.mask = None

    def forward(self, x):
        # 大于零的元素不用动，把小于等于0的元素赋值为0即可
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class AffineLayer:
    """Affine层的正向与反向传播，正向传播输出内积结果，反向传播计算出dw,dx,db，因此需要记录下来

    """

    def __init__(self, w=None, b=None):
        # 一般在正向传播forward时才传入
        self.X = None
        # self.W = np.random.randn(2, 3)
        self.W = w
        self.bias = b
        self.dW = None
        self.db = None

    def forward(self, x):
        self.X = x
        # 正向传播
        y = np.dot(x, self.W) + self.bias
        return y

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.X.T, dout)
        db = np.sum(dout, axis=0)
        self.dW = dW
        self.db = db
        return dx


class SoftmaxLossLayer:
    """ SoftmaxLoss层的正向与反向传播，正向传播输出内积结果，即先经过softmax函数再经过交叉熵函数即为正向传播的结果
    反向传播的结果就是y-t，即正向传播softmax输出的结果-标签
    由此，该节点需要保存的参数为：x, y, loss
    """

    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        y = SimpleLayer.softmax(x)
        self.y = y
        self.t = t
        self.loss = SimpleLayer.cross_entropy_error(y, t)
        return self.loss

    def backward(self, dout=1):
        # 真实标签的个数
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


if __name__ == '__main__':
    net = SoftmaxLossLayer()
    out = net.forward(np.array([[1., -2.3, 1.2], [1.2, -3., 4.1]]), np.array([[1, 0, 0], [0, 1, 0]]))
    dx = net.backward()
    print(dx)
