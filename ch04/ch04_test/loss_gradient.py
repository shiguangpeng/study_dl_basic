import numpy as np

from ch03.ch03_test.simple_layer import SimpleLayer
from ch03.ch03_test.my_gradient import numerical_gradient


def f(W):
    print("进入了f函数")
    return lyr.gradient_loss(x, t)


if __name__ == '__main__':
    x = np.array([[0.6, 0.9]])
    lyr = SimpleLayer(x, 3, activation='softmax')
    # z = lyr.forward()

    lyr.trace_layer_params()
    t = np.array([[0, 1, 0]])

    # 求损失函数对权重weight的偏导
    dW = numerical_gradient(f, lyr.weight)
    print(dW)
