import numpy as np
import matplotlib.pyplot as plt


# 本书示例代码提供的load_mnist函数
from dataset.mnist import load_mnist
# 自己写的拥有2个隐藏层的简单神经网络
from ch04.ch04_test.simple_2lyrs_net import Simple2LayersNet


def main():
    # 下载mnist数据集, x_train有6w条
    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)
    print(x_train.shape)
    # 随时记录loss值
    train_loss_list = []

    # 超参数
    iters_num = 1000
    # 训练样本的数量

    # 只选取300个，故意使其过拟合
    x_train = x_train[:300]
    t_train = t_train[:300]

    train_size = x_train.shape[0]
    # mini-batch的大小
    batch_size = 10
    # 学习率，使用SGD时调整下降的快慢
    lr = 0.1
    # 平均每个epoch的重复次数，batch_size为100，训练集大小为train_size
    # 考虑到整笔一起输入使用max
    # 这个参数的意思是需要iter_per_epoch次后，才是一个epoch
    iter_per_epoch = max(train_size / batch_size, 1)
    # 测试集，验证集上的精度
    train_acc_list = []
    test_acc_list = []

    # 初始化网络
    net = Simple2LayersNet(input_size=x_train.shape[1], hidden_size=100, output_size=10)
    print(net.param['w1'].shape)

    # SGD，使用随机梯度下降算法训练网络
    for i in range(iters_num):
        print('epoch {0} start'.format(i))
        # 选取mini-batch，choice的意思是，从np.arange(train_size)的序列中选出batch_size个，
        # 返回的就是被选中的数字，这个数字恰好就是样本数据集的索引
        batch_mask = np.random.choice(train_size, batch_size)

        # 根据mask在样本中挑选出对应的元素，使用numpy的数组索引
        batch_train = x_train[batch_mask]
        batch_test = t_train[batch_mask]

        # 计算梯度
        grads = net.numerical_gradient(batch_train, batch_test)
        # 梯度下降
        for key in ('w1', 'b1', 'w2', 'b2'):
            net.param[key] -= lr * grads[key]

        loss = net.loss(batch_train, batch_test)
        train_loss_list.append(loss)

        # 经过一定的epoch后在训练集和测试集上验证精度
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc is {0} | test_acc is {1}".format(str(train_acc), str(test_acc)))

    # 训练完成后，画图
    x = np.linspace(0, iters_num, 50)
    y = train_loss_list

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()

