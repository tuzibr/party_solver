import numpy as np
import sympy as sp


def bisection_method(f, a, b, tol=1e-5, max_iter=100):
    """
    使用二分法求解函数 f 在区间 [a, b] 上的零点。

    :param f: 目标函数
    :param a: 区间左端点
    :param b: 区间右端点
    :param tol: 容差
    :param max_iter: 最大迭代次数
    :return: 零点的近似值
    """
    if f(a) * f(b) >= 0:
        raise ValueError("函数在区间两端的值必须异号")

    for _ in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0 or (b - a) / 2 < tol:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2


# 示例函数
def example_function(x):
    return x ** 2 - 2


# 使用二分法求解零点
zero_point = bisection_method(example_function, 1, 2)
print(f"二分法求得的零点: {zero_point}")