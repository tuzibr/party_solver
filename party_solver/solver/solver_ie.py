import numpy as np
import copy
import itertools
from party_solver.solver import solver_bc


def implicit_enumeration(c, A, b, S, B, E, lb, ub, integer_indices, binary_indices):
    best_solution = None
    best_objective = float('inf')
    # print(binary_indices)
    # 深拷贝原始矩阵和向量
    c = copy.deepcopy(c)
    A_new = copy.deepcopy(A)
    b_new = copy.deepcopy(b)
    S_new = copy.deepcopy(S)
    B_new = copy.deepcopy(B)
    E_new = copy.deepcopy(E)


    # 二元矩阵
    A_binary = np.zeros((len(binary_indices), A_new.shape[1]))
    # 根据 binary_indices 在对应的位置置 1
    for i, index in enumerate(binary_indices):
        A_binary[i, index] = 1
    A = np.vstack((A_binary, A_new))

    b_binary = np.zeros(len(binary_indices))
    b = np.concatenate((b_binary, b_new))

    S_binary = np.zeros((len(binary_indices), S_new.shape[1]))
    S = np.vstack((S_binary, S_new))

    B_binary = np.eye(len(binary_indices))
    # print(B_binary)
    # print(B_new)
    # 获取 B_new 的形状
    rows_new, cols_new = B_new.shape
    # 创建一个与 B_new 相同行数、列数为 B_binary 列数加上 B_new 列数的零矩阵
    B = np.zeros((rows_new + len(binary_indices), len(binary_indices) + cols_new))
    # 将 B_binary 插入到 B_new 的右上角
    B[:len(binary_indices), -len(binary_indices):] = B_binary
    # 将 B_new 放置到 B_combined 的左下角
    B[len(binary_indices):, :cols_new] = B_new


    E_binary = np.zeros((len(binary_indices)))
    E = np.concatenate((E_binary, E_new))


    iteration=0
    # 枚举所有二元变量的组合
    for combination in itertools.product([0, 1], repeat=len(binary_indices)):

        # 设置当前组合的二元变量值
        for i, value in enumerate(combination):
            # print(f"Index: {i}, Value: {value}")
            # 更新 lb 和 ub
            b[i] = value
            # print(b)

        # 调用分支定界算法求解
        result = solver_bc.branch_and_bound(c, A, b, S, B, E, lb, ub, integer_indices, best_objective)
        if result:
            solution, objective, _, _, _, _, _, _, _, _ = result
            # print([float(frac) for frac in solution])
            # print(objective)
            # 更新最优解
            iteration += 1
            if objective < best_objective:
                best_solution = solution
                best_objective = objective

    print('implicit_enumeration iteration:', iteration, '')
    return best_solution


if __name__ == '__main__':
    c = np.array([-2.0, -3.0, -4.0, -0.0, -0.0, -0.0])
    A = np.array(
        [[1.0, 0.0, 0.0, -200.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, -200.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, -200.0],
         [-1.0, 0.0, 0.0, 80.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0, 80.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 80.0],
         [1.5, 3.0, 5.0, 0.0, 0.0, 0.0], [280.0, 250.0, 400.0, 0.0, 0.0, 0.0]])
    b = np.array([0, 0, 0, 0, 0, 0, 600, 60000])
    S = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    B = np.array([[], [], [], [], [], [], [], []])
    E = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    lower_bounds = np.array([0, 0, 0, 0, 0, 0])
    upper_bounds = np.array([200, 200, 200, 1, 1, 1])
    integer_indices = [0, 1, 2]
    binary_indices = [3,4,5]
    x = implicit_enumeration(c, A, b, S, B, E, lower_bounds, upper_bounds, integer_indices, binary_indices)
    print('Best solution:', [float(frac) for frac in x])

