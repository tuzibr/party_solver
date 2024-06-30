import sympy as sp
import numpy as np
from party_solver.tools import brent_method
from party_solver.tools import difftools
def line_search(f, x_val, direction, vars):
    alpha = sp.Symbol('alpha')
    f_sub = f.subs(dict(zip(vars, x_val + alpha * direction)))
    f_func = sp.lambdify(alpha, f_sub)
    result = brent_method.brent_method(f_func,tol=1e-5)
    return result

def gradient_descent(f, vars, x0, epsilon=1e-5, max_iter=1000):
    """
    使用梯度下降法优化函数f。

    参数:
    f: 要优化的函数。
    vars: 函数f中的变量列表。
    x0: 初始猜测的解的数组。
    epsilon: 梯度下降停止的阈值，表示解的精度。
    max_iter: 最大迭代次数。

    返回:
    x_val: 梯度下降后的解。
    """
    # 初始化解的值
    x_val = np.array(x0, dtype=float)
    # 计算初始点处的梯度

    # 开始迭代
    for i in range(max_iter):
        grad = difftools.gradient(f, vars, x_val)
        # 梯度下降的方向为梯度的反方向
        direction = -grad
        # 线性搜索确定步长
        alpha = line_search(f, x_val, direction, vars)
        # 计算下一个迭代点
        x_next = x_val + alpha * direction
        # 检查是否达到收敛条件
        if np.linalg.norm(x_next - x_val) < epsilon:
            print(f"gradient descent achieved after {i + 1} iterations")
            break
        # 更新解的值
        x_val = x_next

    return x_val


if __name__ == "__main__":

    x, y, z = sp.symbols('x y z')

    f_expr = x ** 2 + 25* y ** 2

    x0 = [1, 1, 1.0]

    optimal_x = gradient_descent(f_expr, (x, y, z), x0)
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_expr.subs({x: optimal_x[0], y: optimal_x[1], z: optimal_x[2]}))
