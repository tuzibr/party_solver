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

def quasi_newton_optimization(f, vars, x0, epsilon=1e-5, max_iter=1000, method = 'bfgs'):
    """
    使用拟牛顿法进行优化。
    参数:
    f: 待优化的函数。
    x0: 初始猜测值。
    epsilon: 收敛精度，默认为1e-5。
    max_iter: 最大迭代次数，默认为1000。
    method: 使用的拟牛顿法方法，可选'bfgs'或'dfp'，默认为'bfgs'。
    返回:
    最优解x_val。
    """
    # 初始化近似海塞矩阵的逆为单位阵
    H = np.eye(len(vars))
    x_val = x0
    for i in range(max_iter):

        # 计算梯度
        grad = difftools.gradient(f, vars, x_val)
        # 搜索方向为海塞矩阵逆乘梯度
        direction = -np.dot(H, grad)
        # 线搜索，确定下一步的步长
        alpha = line_search(f, x_val, direction, vars)
        # 计算下一个迭代点
        x_next = x_val + alpha * direction

        if np.linalg.norm(x_next - x_val) < epsilon:
            print(f"Quasi-Newton optimization convergence achieved after {i} iterations")
            break

        # 计算下一个迭代点的梯度
        grad_next = difftools.gradient(f, vars, x_next)

        if method =='bfgs':
            # bfgs方法近似海塞矩阵的逆
            s = alpha * direction
            y = grad_next - grad
            delta = np.dot(y, s)
            if delta >0:
                rho = 1.0 / delta
                I = np.eye(len(vars))
                H = (I - rho * np.outer(s, y)).dot(H).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        elif method == 'dfp':
            # dfp方法近似海塞矩阵的逆
            s = alpha * direction
            y = grad_next - grad
            delta = np.dot(y, s)
            if delta >0:
                rho = 1 / delta
                Hs = np.dot(H, s)
                H += rho * np.outer(s, s) - np.outer(Hs, Hs) / np.dot(s, Hs)

        else:
            raise ValueError("Invalid method specified")
        x_val = x_next
    return x_val



if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    f_expr = x ** 2 + y ** 2 + z**2 + 2 * (x * y)**2 + 3 * (y * z)**2
    x0 = np.array([1.0, 1.0, 1.0])

    optimal_x = quasi_newton_optimization(f_expr, (x, y, z), x0, method='bfgs')
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_expr.subs({x: optimal_x[0], y: optimal_x[1], z: optimal_x[2]}))
