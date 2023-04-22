import numpy as np


def AND(x1, x2):
    w1, w2, b = 0.5, 0.5, 0.7
    y = np.dot(x1, w1) + np.dot(x2, w2)
    if y - b > 0:
        return 1
    else:
        return 0

def NAND(x1, x2):
    w1, w2, b = 0.5, 0.5, 0.7
    y = np.dot(x1, w1) + np.dot(x2, w2)
    if y - b > 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    w1, w2, b = 0.8, 0.9, 0.7
    x = np.array([x1, x2])
    w = np.array([w1, w2])
    result = np.sum(w*x)+b
    if result - b > 0:
        return 1
    else:
        return 0


res = OR(0, 0)
res2 = OR(0, 1)
res3 = OR(1, 1)
res4 = OR(1, 0)
print(res, res2, res3, res4)


# res = AND(0, 0)
# res2 = AND(0, 1)
# res3 = AND(1, 1)
# res4 = AND(1, 0)
# print(res, res2, res3, res4)
# # results: 0 0 1 0

# res = NAND(0, 0)
# res2 = NAND(0, 1)
# res3 = NAND(1, 1)
# res4 = NAND(1, 0)
# print(res, res2, res3, res4)
# # results: 0 0 1 0



