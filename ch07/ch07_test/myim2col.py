import numpy as np
from common import util
from common.util import im2col


def my_im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data: array-like
                输入的数据，图片矩阵
    filter_h : int
                图片的高
    filter_w : int
                图片的宽
    stride : int
                滤波器移动的步长
    pad : int
                填充的大小

    Returns
    -------
    col : array-like
            返回的是转化后的矩阵（数组），即经过卷积的结果（特征图）

    """

    # 读取输入图片集合的形状，按照规范（图片数量，通道， 高，宽）
    global col
    # 计算输出的特征图大小
    N, C, H, W = input_data.shape
    # 先计算输出的宽高
    ow = (W + 2*pad - filter_w) // stride + 1
    oh = (H + 2*pad - filter_h) // stride + 1

    # 填充高和宽
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    # 初始化输出的矩阵形状
    col = np.zeros((ow*oh*N, C*filter_w*filter_h))

    count = 0
    for i in range(ow):
        count += 1
        for j in range(oh):
            # x, y的最大值都不能超过原来长宽的最大值
            # 以步长stride在矩阵中搜索长度为i:i+filter_h，宽度为j:j+filter_w的大小的矩阵
            # 按横向拉平矩阵，变为一维
            p = 2*(i+j)
            q = p+2
            if count != 1:
                p += 2
                q = p+2
            col[p: q, :] = img[:, :, i: i+filter_h:stride, j:j+filter_w:stride].reshape(-1, C*filter_w*filter_h)
    # return util.im2col(input_data, filter_h, filter_w)
    return col


if __name__ == '__main__':
    data = np.random.rand(2, 2, 4, 4)
    result = my_im2col(data, 3, 3)
    result2 = im2col(data, 3, 3, 1, 0)
    print("xxx")
