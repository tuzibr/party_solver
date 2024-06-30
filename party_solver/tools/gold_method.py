import math
import matplotlib.pyplot as plt
import sympy as sp

def golden_section_search(f, a, b, tol=1e-5):
    g = (math.sqrt(5) - 1) / 2  # 黄金比例

    c = b - g * (b - a)
    d = a + g * (b - a)

    fc = f(c)
    fd = f(d)

    # 记录每次迭代的区间
    intervals = [(a, b)]

    while abs((b - a)/b) > tol:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + g * (b - a)
            fd = f(d)

        intervals.append((a, b))


    result = (a + b) / 2

    return result, intervals


if __name__ == '__main__':

    x = sp.Symbol('x')

    objective_function_sympy = sp.cos(x)

    objective_function = sp.lambdify(x, objective_function_sympy, 'math')

    # 定义搜索区间和容许误差
    a = 0  # 起始区间
    b = 2 * math.pi  # 结束区间
    tol = 0.1  # 容许误差

    # 使用黄金分割法寻找目标函数的最小值
    min_x, intervals = golden_section_search(objective_function, a, b, tol)
    min_value = objective_function(min_x)

    print(f"min_fun = {min_x}, min_value = {min_value}")

    # 绘制函数曲线
    x_values = [x * 0.01 for x in range(int(2 * math.pi * 100) + 1)]
    y_values = [objective_function(x) for x in x_values]

    plt.plot(x_values, y_values, label='function')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # 绘制搜索区间的变化
    for (a, b) in intervals:
        plt.plot([a, b], [objective_function(a), objective_function(b)], 'ro-')

    plt.title('Golden Section Search')
    plt.xlabel('x')
    plt.ylabel('function value')
    plt.legend()
    plt.show()
