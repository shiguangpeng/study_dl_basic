import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist


def load_pkl(file_path: str):
    res = None
    with open(file_path, mode='rb') as fb:
        res = pkl.load(fb)
    return res


# 从数据集中随机抽取mini-batch（训练集的数据与标签，以mnist手写数字数据集为例）

def random_choice():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=False)
    # print(x_train.shape)
    # print(t_train.shape)
    # print(x_test.shape)
    print(t_train)
    # 关键就是使用choice在训练集中选取batch_size大小的样本
    train_size = x_train.shape[0]
    batch_size = 10
    # 从0 到 train_size-1中随机抽取batch_size大小的样本的索引，batch_size是一个数组
    batch_mask = np.random.choice(train_size, batch_size)
    # 根据索引数组取出对应的元素
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    return x_batch, t_batch


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7))


def mini_batch_cer(y, t):
    # 判断是不是只有一个元素，二维表示法，但只有一个元素
    if y.ndim == 1:
        # 去掉外层的中括号，相当于将数据从二维数组中取出
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 若监督/标签数据是one-hot编码，需要转成真实的标签值
    if y.size == t.size:
        t = t.argmax(t)

    batch_size = y.shape[0]
    a = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    return a


def func(x):
    return 2 * x ** 2 + 1


#
def func2(x):
    x2 = 0.0
    if len(x.shape) == 2:
        x2 = x[0].copy()
    return (1/20) * x2[0] ** 2 + x2[1] ** 2


def numerical_diff(f, x):
    # 微小的步长
    h = 1e-4
    return (f(x + h) - f(x - h)) / 2 * h


def my_function(x):
    return np.sum(x * x)


def numerical_gradient(f, x_val):
    '''数值微分方法计算偏导，这不是利用反向传播

    Parameters
    ----------
    f: 损失函数
    x_val: 需要计算导数的自变量，这个值的修改会影响外层作用域的值，因为是引用传递，

    Returns:
    -------

    '''
    h = 1e-4
    grad = np.zeros_like(x_val)
    if len(x_val.shape) == 1:
        # 求当前变量的梯度
        for idx in range(x_val.shape[0]):
            val = x_val[idx]
            f_1 = f(val + h)
            f_2 = f(val - h)
            x_val[idx] = val
            grad[idx] = (f_1 - f_2) / (2 * h)
    else:
        for idx in range(x_val.shape[0]):
            for jdx in range(x_val.shape[1]):
                val = x_val[idx][jdx]
                x_val[idx][jdx] = val + h
                # 交叉熵的计算仅与正确标签处的概率有关，因此在计算不同的w的导数时，只需要在对应的w（如w11）上使用导数的定义即可
                f_1 = f(x_val)
                x_val[idx][jdx] = val - h
                f_2 = f(x_val)
                grad[idx][jdx] = (f_1 - f_2) / (2 * h)
                # 计算完导数后，把对应元素的值还原，避免影响下一个元素的交叉熵。
                x_val[idx][jdx] = val
    return grad


def sgd(f, x: np.ndarray, times: int = 50, lr: float = 0.1):
    """This function is SGD algorithm by using numpy

    Parameters:
        f: 多元函数
        x: 自变量取值列表
        times: 迭代次数
        lr: learning rate 学习率

    Returns:
        result: 使用梯度下降算法得出的近似值

    Raises:
        DivideByZero: 除0异常
    """

    # 使用numpy解决：使用数值法求梯度，并使用梯度下降法求函数f的最小值。
    #
    x_list = []
    x_init = x.copy()
    # 使用梯度下降法反复迭代指定的次数
    for i in range(times):
        x_init = x_init - lr * numerical_gradient(f, x_init)
        # print(x_init[0])
        x_list.append(x_init[0])
    return x_list


if __name__ == '__main__':
    # 测试sample_weight.pkl中的内容是什么，是字典，网络权重和偏置的字典，共4层（包括输出层）
    # filepath = r'../sample_weight.pkl'
    # for key, item in load_pkl(filepath).items():
    #     print(key, item.shape)
    # y = np.array([0.3, 0.02, 0.05, 0.03, 0.1, 0.1, 0.1, 0.2, 0.05, 0.05],
    #              [0.1, 0.02, 0.05, 0.03, 0.3, 0.1, 0.1, 0.2, 0.05, 0.05])
    # t = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    # # result = mini_batch_cer(y, t)
    # print(y[[0, 1], t])

    # random_choice()
    # x = 1
    # cc = numerical_diff(func, x, 1e4)

    # result = numerical_gradient(my_function, np.array([1., 2.]))
    # result2 = sgd(my_function, np.array([-3, 4]), 50, 0.1)
    # print(result2)

    # 选择合适的学习率，才能成功进行梯度下降
    result3 = sgd(func2, np.array([[-3., 4.]]), 1000, 0.1)
    print(result3)
    x = []
    y = []
    for coords in result3:
        x.append(coords[0])
        y.append(coords[1])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()
