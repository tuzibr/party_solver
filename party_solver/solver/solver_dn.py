import sympy as sp
import numpy as np
from party_solver.tools import brent_method
from party_solver.tools import difftools


def line_search(f, x_val, direction, vars):
    alpha = sp.Symbol('alpha')
    f_sub = f.subs(dict(zip(vars, x_val + alpha * direction)))
    f_func = sp.lambdify(alpha, f_sub)
    result = brent_method.brent_method(f_func, tol=1e-5)

    return float(result)

def damped_newton_optimization(f, vars, x0, epsilon=1e-5, max_iter=1000):
    """
    使用阻尼牛顿法进行优化。
    参数:
    f: 待优化的函数。
    x0: 初始猜测值。
    epsilon: 收敛精度，默认为1e-5。
    max_iter: 最大迭代次数，默认为1000。
    返回:
    最优解x_val。
    """

    x_val = x0
    for i in range(max_iter):
        H = difftools.hessian(f, vars, x_val)
        # 计算梯度
        grad = difftools.gradient(f, vars, x_val)
        # 搜索方向为海塞矩阵逆乘梯度
        direction = -np.dot(H, grad)
        # 线搜索，确定下一步的步长
        alpha = line_search(f, x_val, direction, vars)
        # 计算下一个迭代点
        x_next = x_val + alpha * direction
        if np.linalg.norm(x_next - x_val) < epsilon:
            print(f"Damped-Newton optimization convergence achieved after {i} iterations")
            break
        x_val = x_next
    return x_val



if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    f_expr = x ** 2 + y ** 2 + z**2 + 2 * (x * y)**2 + 3 * (y * z)**2
    x0 = np.array([1.0, 1.0, 1.0])

    optimal_x = damped_newton_optimization(f_expr, (x, y, z), x0)
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_expr.subs({x: optimal_x[0], y: optimal_x[1], z: optimal_x[2]}))
