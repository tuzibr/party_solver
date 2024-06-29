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

def conjugate_gradient_optimization(f, vars, x0, epsilon=1e-5, max_iter=1000):
    """
    使用共轭梯度法进行优化。

    参数:
    f: 待优化的目标函数。
    vars: 函数f中的变量列表。
    x0: 初始猜测的解。
    epsilon: 收敛的精度阈值。
    max_iter: 最大迭代次数。

    返回:
    x_val: 优化后的解。
    """
    # 初始化解的值
    x_val = np.array(x0, dtype=float)
    # 计算初始点的梯度
    # grad = np.array([f.diff(var).subs(dict(zip(vars, x_val))) for var in vars], dtype=float)
    grad = difftools.gradient(f, vars, x_val)
    # 初始化下降方向为负梯度方向
    direction = -grad
    for i in range(max_iter):

        # 线搜索，确定下一步的步长
        alpha = line_search(f, x_val, direction, vars)
        # 计算下一个迭代点
        x_next = x_val + alpha * direction
        # 检查收敛条件
        if np.linalg.norm(x_next - x_val) < epsilon:
            print(f"conjugate-gradient convergence achieved after {i + 1} iterations")
            break

        # 计算下一个迭代点的梯度
        grad_next = difftools.gradient(f, vars, x_val)
        # 计算共轭梯度法中的beta值
        # 生成共轭梯度
        beta = np.dot(grad_next, grad_next) / np.dot(grad, grad)
        # 更新下降方向
        direction = -grad_next + beta * direction
        # 更新解的值和梯度
        x_val = x_next
        grad = grad_next

    return x_val


if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    f_expr = x**2 + y**2 + z**2 + sp.cos(x)
    x0 = np.array([1.0, 1.0, 1.0])

    optimal_x = conjugate_gradient_optimization(f_expr, (x, y, z), x0)
    print("Optimal solution:", optimal_x)
    print("Objective function value:", f_expr.subs({x: optimal_x[0], y: optimal_x[1], z: optimal_x[2]}))
