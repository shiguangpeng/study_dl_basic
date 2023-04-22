import numpy as np

from ch05.ch05_test.MyBasicLayer import MyMulLayer, MyAddLayer


def main():
    """反向传播链上有5个变量，分别是苹果个数、苹果单价，橘子个数、橘子单价，消费税

    Returns 梯度
    -------

    """

    # 初始化输入
    apple_cout = 2
    apple_price = 100
    orange_cout = 3
    orange_price = 150
    customer_tax = 1.1

    # 新建4个节点，3个乘法节点，1个加法节点
    apple_mul_layer = MyMulLayer()
    orange_mul_layer = MyMulLayer()
    apple_orange_add_layer = MyAddLayer()
    taxes_mul_layer = MyMulLayer()

    # 按照计算图进行正向传播
    apple_total_price = apple_mul_layer.forward(apple_cout, apple_price)
    orange_total_price = orange_mul_layer.forward(orange_cout, orange_price)
    apple_orange_sum_price = apple_orange_add_layer.forward(apple_total_price, orange_total_price)
    taxes_price = taxes_mul_layer.forward(apple_orange_sum_price, customer_tax)

    print(apple_total_price, orange_total_price, apple_orange_sum_price, taxes_price)

    # 反向传播，最后一层向后传播时的导数为 1
    d_sum_of_apple_orange, d_taxes = taxes_mul_layer.backward(1)
    print("最后一层乘法节点向前传播的结果：\n{0}, {1}".format(d_sum_of_apple_orange, d_taxes))
    d_apple_sum_prices, d_orange_sum_prices = apple_orange_add_layer.backward(d_sum_of_apple_orange)
    print("倒数第2层乘法节点向前传播的结果：\n{0}, {1}".format(d_apple_sum_prices, d_orange_sum_prices))
    d_apple_count, d_apple_price = apple_mul_layer.backward(d_apple_sum_prices)
    print("倒数第3层乘法节点向前传播的结果：\n{0}, {1}".format(d_apple_count, d_apple_price))
    d_orange_count, d_orange_price = orange_mul_layer.backward(d_orange_sum_prices)
    print(d_orange_count, d_orange_price)


if __name__ == '__main__':
    main()
