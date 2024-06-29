import numpy as np
from sympy import apply_finite_diff, symbols, lambdify
import copy

def gradient(f, vars, x0, h=1e-5):
    """
    计算给定函数f关于变量vars在点x0处的梯度。

    参数:
    f: 待求导的函数表达式。
    vars: 函数f中的自变量列表。
    x0: 函数f求导的初始点，即x0处的梯度。
    h: 求导的步长，用于有限差分近似导数，默认值为1e-5。

    返回:
    梯度向量，以numpy数组形式表示。
    """
    # 深拷贝x0以避免修改原始数据
    x_diff = copy.deepcopy(x0)
    # 将元组转换为列表，以便进行修改
    x_diff = tuple(list(x_diff))
    # 使用lambdify将 sympy 表达式转换为可调用的函数
    f = lambdify(vars, f)
    # 初始化梯度向量
    grad = []
    # 遍历每个变量，计算其导数
    for i, var in enumerate(vars):
        # 构建变量的微小变化范围，用于有限差分
        var_list = [x_diff[i] - h, x_diff[i], x_diff[i] + h]
        # 计算函数在每个变量值下的值
        f_values = [f(*(x_diff[:i] + (var_val,) + x_diff[i+1:])) for var_val in var_list]
        # print(f_values)
        # 应用有限差分公式计算导数
        df_dvar = apply_finite_diff(1, var_list, f_values, x_diff[i])
        # 将导数添加到梯度向量
        grad.append(df_dvar)
    # 将梯度向量转换为numpy数组，并指定数据类型为浮点数
    return np.array(grad, dtype=float)

def jacobian(F, vars, x0, h=1e-5):
    """
    计算给定向量值函数F关于变量vars在点x0处的雅各布矩阵。

    参数:
    F: 待求导的向量值函数表达式列表。
    vars: 向量值函数F中的自变量列表。
    x0: 向量值函数F求导的初始点，即x0处的雅各布矩阵。
    h: 求导的步长，用于有限差分近似导数，默认值为1e-5。

    返回:
    雅各布矩阵，以numpy数组形式表示。
    """
    # 深拷贝x0以避免修改原始数据
    x_diff = copy.deepcopy(x0)
    # 将元组转换为列表，以便进行修改
    x_diff = tuple(list(x_diff))
    if len(F)==0:
        return np.array([])
    elif not isinstance(F[0], type(lambda: None)):
        # 使用lambdify将 sympy 表达式转换为可调用的函数
        F_funcs = [lambdify(vars, f) for f in F]
    else:
        F_funcs = F

    # 初始化雅各布矩阵
    m = len(F)
    n = len(vars)
    jacobian_matrix = np.zeros((m, n))
    # 遍历每一对函数和变量，计算偏导数
    for i, f_func in enumerate(F_funcs):
        for j, var in enumerate(vars):
            # 构建变量的微小变化范围，用于有限差分
            var_list = [x_diff[j] - h, x_diff[j], x_diff[j] + h]
            # 计算函数在每个变量值下的值
            f_values = [f_func(*(x_diff[:j] + (var_val,) + x_diff[j+1:])) for var_val in var_list]
            # 应用有限差分公式计算导数
            df_dvar = (f_values[2] - f_values[0]) / (2 * h)
            # 将导数添加到雅各布矩阵
            jacobian_matrix[i, j] = df_dvar
    # 返回雅各布矩阵
    return np.array(jacobian_matrix, dtype=float)

def hessian(f, vars, x0, h=1e-5):
    """
    计算给定函数f关于变量vars在点x0处的海塞矩阵。

    参数:
    f: 待求导的函数表达式。
    vars: 函数f中的自变量列表。
    x0: 函数f求导的初始点，即x0处的海塞矩阵。
    h: 求导的步长，用于有限差分近似导数，默认值为1e-5。

    返回:
    海塞矩阵，以numpy数组形式表示。
    """
    # 深拷贝x0以避免修改原始数据
    x_diff = copy.deepcopy(x0)
    # 将元组转换为列表，以便进行修改
    x_diff = tuple(list(x_diff))
    # 使用lambdify将 sympy 表达式转换为可调用的函数
    f = lambdify(vars, f)
    # 初始化海塞矩阵
    n = len(vars)
    hessian_matrix = np.zeros((n, n))
    # 遍历每一对变量，计算二阶导数
    for i in range(n):
        for j in range(i, n):
            # 构建变量的微小变化范围，用于有限差分
            var_list_i = [x_diff[i] - h, x_diff[i], x_diff[i] + h]
            var_list_j = [x_diff[j] - h, x_diff[j], x_diff[j] + h]
            f_values = []
            for vi in var_list_i:
                for vj in var_list_j:
                    x_ij = list(x_diff)
                    x_ij[i] = vi
                    x_ij[j] = vj
                    f_values.append(f(*x_ij))
            f_values = np.array(f_values).reshape((3, 3))
            if i == j:
                d2f_dvaridvarj = apply_finite_diff(2, var_list_i, f_values[1], x_diff[i])
            else:
                d2f_dvaridvarj = apply_finite_diff(1, var_list_i, apply_finite_diff(1, var_list_j, f_values, x_diff[j]), x_diff[i])

            # 将二阶导数添加到海塞矩阵
            hessian_matrix[i, j] = d2f_dvaridvarj
            if i != j:
                hessian_matrix[j, i] = d2f_dvaridvarj
    # 返回海塞矩阵
    return np.array(hessian_matrix, dtype=float)



if __name__ == '__main__':
    x, y = symbols('x y')

    f = x**3 + y**3 +x*y

    x0, y0 = 1, 3

    h = 1e-6

    # Compute the gradient
    gradient = gradient(f, (x, y), (x0, y0), h)
    print(gradient)

    hessian_matrix = hessian(f, (x, y), (x0, y0), h)
    print(hessian_matrix)

    f = [f, 2*f, 1+f*2, 2*f*2]
    jacobian_matrix = jacobian(f, (x, y), (x0, y0), h)
    print(jacobian_matrix)