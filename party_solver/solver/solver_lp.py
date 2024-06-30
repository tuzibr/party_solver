import numpy as np
from fractions import Fraction
import copy


def elementary_row_operations(matrix, basis_indices):
    # 获取行数
    num_rows = matrix.shape[0]
    # 初等行变换
    for index in basis_indices:

        pivot_row = None
        # 找主元
        for row in range(num_rows):
            if matrix[row, index] != 0:
                pivot_row = row
                break

        if pivot_row is None:
            continue

        # 主元化为1
        pivot = matrix[pivot_row, index]
        # print('zhuyuan',matrix[pivot_row])
        matrix[pivot_row] = matrix[pivot_row] / pivot

        # 除主元行为0
        for row in range(num_rows):
            if row != pivot_row:
                factor = matrix[row, index]
                matrix[row] = matrix[row] - factor * matrix[pivot_row]

    return matrix


# def elementary_row_operations_for_dual(matrix, basis_indices):
#     # 获取行数
#     num_rows = matrix.shape[0]
#
#     # 初等行变换
#     for index in basis_indices[:-1]:
#         # print(index)
#         pivot_row = None
#         # 寻找主元行
#         for row in range(num_rows):
#             if matrix[row, index] != 0:
#                 pivot_row = row
#                 break
#
#         if pivot_row is None:
#             continue
#         # print(pivot_row)
#         # 主元化为1
#         pivot = matrix[pivot_row, index]
#         # print('zhuyuan',matrix[pivot_row,-1])
#         matrix[pivot_row] = matrix[pivot_row] / pivot
#
#         # 除主元行为0
#         row = num_rows-2
#         factor = matrix[row, index]
#         matrix[row] = matrix[row] - factor * matrix[pivot_row]
#         # print('huajian',matrix[-2])
#
#     return matrix

def pivot_finder(matrix, idx):
    # 获取行数
    num_rows = matrix.shape[0]
    pivot_row = None
    for row in range(num_rows):
        if matrix[row, idx] != 0:
            pivot_row = row
            break

    return pivot_row


def simplex_method_high_precision(d, s, c):
    # print('high')
    # 将数组 d 转换为 Fraction 类型


    d = np.array(d, dtype=object)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i, j] = Fraction(str(d[i, j]))  # 使用字符串转换来保证精度
    (bn, cn) = d.shape  # 获取行数和列数

    while min(d[-1][:-1]) < 0:  # 转入条件变为查找最小的负数
        jnum = np.argmin(d[-1][:-1])  # 转入下标
        column = d[:-1, jnum]

        if all(c <= 0 for c in column):  # 检查是否所有元素都不是正的（没有上界）
            return None  # 解无界

        ratios = []
        for i in range(bn - 1):
            if column[i] > 0:
                ratios.append(d[i, -1] / column[i])
            else:
                ratios.append(Fraction(10 ** 10))  # 使用一个大数来代替无穷大


        # 将数组反转
        reversed_ratios = np.flip(ratios)

        # 使用 np.argmin() 找到最小值在反转后数组中的索引
        index_of_min_reversed = np.argmin(reversed_ratios)

        # 计算从后往前的索引
        inum = len(ratios) - index_of_min_reversed - 1


        s[inum] = jnum  # 更新基变量
        pivot = d[inum, jnum]
        d[inum] /= pivot
        for i in range(bn):
            if i != inum:
                d[i] -= d[i][jnum] * d[inum]

    num_decision_vars = c.size

    # 创建结果数组
    x_values = np.zeros(num_decision_vars, dtype=object)
    for i in range(num_decision_vars):
        if i in s:
            x_values[i] = d[pivot_finder(d, i), -1]
        else:
            x_values[i] = Fraction(0)

    # 将 Fraction 转换为浮点数
    # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
    return x_values, s, np.sum(c * [float(value) for value in x_values]).astype(float), d


def simplex_method_normal_precision(d, s, c, original_d=None):
    if original_d is None:
        original_d = d.copy()  # 备份初始矩阵
    (bn, cn) = d.shape  # 获取行数和列数

    iteration_count = 0  # 初始化迭代计数器

    # print(d)

    while min(d[-1][:-1]) < 0:  # 转入条件变为查找最小的负数
        iteration_count += 1  # 每次迭代时增加计数
        jnum = np.argmin(d[-1][:-1])  # 转入下标
        # print('转入下标',jnum)
        column = d[:-1, jnum]
        # print('转入列值',column)

        if np.all(column <= 0):  # 检查是否所有元素都不是正的（没有上界）
            return None  # 解无界

        out = d[:-1, -1] / np.maximum(1e-18, column)  # 用非常小的正数避免除以零
        # print('最小比率',out)
        np.place(out, column <= 0, np.inf)  # 替换无效的比率为无穷大
        inum = np.argmin(out)  # 转出下标
        # print('转出下标',inum)
        s[inum] = jnum  # 更新基变量
        # print('新基变量',s)
        pivot = d[inum, jnum]
        d[inum] /= pivot
        for i in range(bn):
            if i != inum:
                d[i] -= d[i][jnum] * d[inum]
        # print(d)
    # print(f"Total iterations: {iteration_count}")
    # print(d[-1][:-1])
    # print(s)
    # 对偶检查解的可行性
    # 构建基矩阵 B

    B = original_d[:-1, s]  # 注意，这里确保不包括目标函数行
    # print(d)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print('stigular')
        return None
    # 基变量和非基变量的目标函数系数
    c_B = original_d[-1, s]
    non_basic_vars = [i for i in range(cn - 1) if i not in s]
    b = original_d[:-1, -1]
    # 计算基变量的取值 x_B
    x_B = np.dot(B_inv, b)
    # print(x_B)
    if np.any(x_B < 0 - 1e-8):
        return None

    num_decision_vars = c.size
    # 创建一个全为零的解集数组
    x_values = np.zeros(num_decision_vars)
    # 遍历基变量索引 s 和对应的 x_B 值
    for index, value in zip(s, x_B):
        if index < num_decision_vars:  # 确保索引在非松弛变量的范围内
            x_values[index] = value
    # print(x_values)
    # 计算对偶差

    duality_feasible = True
    for j in non_basic_vars:
        c_N = original_d[-1, j]  # 第 j 个非基变量的目标函数系数
        reduced_cost = c_N - np.dot(c_B, B_inv.dot(original_d[:-1, j]))
        if reduced_cost < 0:
            duality_feasible = False
            break
    if duality_feasible:
        # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
        return x_values, s, np.sum(c * [float(value) for value in x_values]).astype(float), d
    else:
        return None


def simplex_method_no_bound(c, A, b, S, B, E):
    # 创建单纯型表
    i = len(c)  # c的长度，即决策变量的数量
    j = A.shape[0]  # A的行数，即约束的数量
    S_size = S.shape[1]
    B_size = B.shape[1]
    # print('columns:', i + S_size + B_size, 'row:', j)
    if B.size == 0:
        # 初始化矩阵
        tableau = np.zeros((j + 1, i + j + 1))  # +1 行目标函数，+j 松弛变量，+1 最后一列为右侧值

        # 填充约束条件
        tableau[:j, :i] = A  # 填充系数矩阵A
        tableau[:j, i:i + j] = np.eye(j)  # 填充松弛变量的单位矩阵
        tableau[:j, -1] = b  # 填充右侧值b

        # 填充目标函数
        tableau[-1, :i] = c
        s = list(range(i, i + j))  # 基变量列表
        backup = np.copy(tableau)

        # 普通精度求解
        solution = simplex_method_normal_precision(tableau, s, c)
        solution = None
        if solution is None:
            # print('nonenonenone')
            solution = simplex_method_high_precision(backup, s, c)
            if solution is None:
                return None,tableau,s
            else:
                # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
                return solution[0], solution[2], solution[1], solution[3]
        # 检查约束
        constraint_satisfied = np.all(np.dot(A, solution[0]) <= b + 1e-8)
        if not constraint_satisfied:
            solution = simplex_method_high_precision(backup, s, c)
        # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
        return solution[0], solution[2], solution[1], solution[3]



    # 两阶段法
    else:
        # 初始化矩阵
        tableau = np.zeros((j + 1, i + S_size + B_size + 1))  # +1 行目标函数，+j 松弛变量，+1 最后一列为右侧值
        # 填充约束条件
        tableau[:j, :i] = A  # 填充系数矩阵A
        tableau[:j, i:i + S_size] = S  # 填充松弛变量的矩阵
        tableau[:j, -1] = b  # 填充右侧值b
        tableau[:j, i + S_size:i + S_size + B_size] = B  # 填充人工变量的矩阵

        original_d = np.zeros((j + 1, i + S_size + 1))  # +1 行目标函数，+j 松弛变量，+1 最后一列为右侧值
        # 填充约束条件
        original_d[:j, :i] = A  # 填充系数矩阵A
        original_d[:j, i:i + S_size] = S  # 填充松弛变量的矩阵
        original_d[:j, -1] = b  # 填充右侧值b
        original_d[-1, :i] = c

        # 第一阶段
        # 填充仅有人工变量目标函数
        tableau[-1, i + S_size:i + S_size + B_size] = 1

        backup = np.copy(tableau)

        c1 = np.zeros(i + S_size + B_size + 1)
        c1[i + S_size:i + S_size + B_size] = 1
        # print('c1',c1)

        # 获取矩阵的行数和列数
        rows, cols = tableau.shape

        # 要检查的总行数减去最后一行
        rows_to_check = rows - 1

        # 初始化记录符合条件的列索引的列表
        s = []

        # 从右向左遍历列，跳过最后一列
        for k in range(0, rows_to_check):
            for col in range(cols - 2, -1, -1):
                # 检查当前列除了最后一行的所有行元素是否都大于0
                if np.any(tableau[k, col] >= 0.9):
                    s.append(col)
                    break
            # 如果找到的符合条件的列数等于总行数减一时停止
            if len(s) == rows_to_check:
                break
        backups = copy.deepcopy(s)
        #
        #
        # elementary_row_operations(tableau, s)
        #
        # s1 = simplex_method_normal_precision(tableau, s, c1)
        #
        # # 假设 i 和 S_size 已经在此处定义
        # if s1 is not None:
        #     if s1[2] > 1e-8 or s1[2] < -1e-8:
        #         return None
        s1 = None
        # 第一阶段不满足直接调用高精度尝试求解
        if s1 is None:
            # print(backups)
            elementary_row_operations(backup, backups)
            s1 = simplex_method_high_precision(backup, backups, c1)
            # print(s1)
            if s1 is None:
                return None
            if s1 is not None:
                # print(s1[0])
                # if set(s1[1]).intersection(set([z for z in range(i + S_size,i + S_size + B_size)])) :
                if s1[2] != 0:
                    return None
                # 第二阶段
                backup = np.delete(s1[3], np.s_[i + S_size:i + S_size + B_size], axis=1)
                backup[-1, :] = 0
                backup[-1, :i] = c


                elementary_row_operations(backup, s1[1])
                solution = simplex_method_high_precision(backup, s1[1], c)
                if solution is None:
                    return None
                else:
                    # 检查约束
                    constraint_satisfied = np.all(E * np.dot(A, solution[0]) <= E * b + 1e-8)
                    # print(constraint_satisfied)
                    if not constraint_satisfied:
                        return None
                    # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
                    return solution[0], solution[2], solution[1], solution[3]

        # 第二阶段
        # 填充原始目标函数
        # print(222222222222)
        # print(tableau)
        tableau = np.delete(tableau, np.s_[i + S_size:i + S_size + B_size], axis=1)

        tableau[-1, :] = 0
        # print(i)
        # print(c)
        tableau[-1, :i] = c

        # print(tableau)
        # print(s1[1])
        # print('bianhuan',tableau,s1[1])
        elementary_row_operations(tableau, s1[1])
        # print(tableau)
        solution = simplex_method_normal_precision(tableau, s1[1], c, original_d)
        # print(solution)
        # 调用高精度尝试求解
        if solution is None:
            # print('nonenonenone')
            elementary_row_operations(backup, backups)
            s1 = simplex_method_high_precision(backup, backups, c1)

            if s1 is None:
                return None
            if s1 is not None:
                # print(s1[2])
                if s1[2] > 1e-8 or s1[2] < -1e-8:

                    return None
                # 第二阶段
                backup = np.delete(s1[3], np.s_[i + S_size:i + S_size + B_size], axis=1)
                backup[-1, :] = 0
                backup[-1, :i] = c
                elementary_row_operations(backup, s1[1])
                solution = simplex_method_high_precision(backup, s1[1], c)
                if solution is None:
                    return None
                else:
                    # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
                    return solution[0], solution[2], solution[1], solution[3]

        # 检查约束
        constraint_satisfied = np.all(E * np.dot(A, solution[0]) <= E * b + 1e-8)

        if not constraint_satisfied:
            elementary_row_operations(backup, backups)
            s1 = simplex_method_high_precision(backup, backups, c1)

            if s1 is None:
                return None
            if s1 is not None:
                # print(s1[2])
                if s1[2] > 1e-8 or s1[2] < -1e-8:

                    return None
                # 第二阶段
                backup = np.delete(s1[3], np.s_[i + S_size:i + S_size + B_size], axis=1)
                backup[-1, :] = 0
                backup[-1, :i] = c
                elementary_row_operations(backup, s1[1])
                solution = simplex_method_high_precision(backup, s1[1], c)
                if solution is None:
                    return None
                else:
                    # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
                    return solution[0], solution[2], solution[1], solution[3]
        else:
            # 返回0，变量的值，1，基变量，2，目标函数值，3，解矩阵
            return solution[0], solution[2], solution[1], solution[3]


def dual_simplex_method(d, s, c):

    d = np.array(d, dtype=object)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            d[i, j] = Fraction(d[i, j])

    if min(d[:, -1][:-1]) >= 0:
        # print(min(d[:, -1][:-1]))
        # print(d[-2,-1])
        raise ValueError("The dual problem is unbounded.")

    if any(d[-1, :-1] < 0):
        print(11111111111111111)


    # print('检验数',d[-1, :-1])
    # 对偶单纯形法 用于迭代增加函数上下界
    while min(d[:, -1][:-1]) < 0:  # 转入条件变为查找最小的负数
        # print('DUAL',d)
        inum = np.argmin(d[:, -1][:-1])  # 转出下标
        # print('转出下标',inum)
        row = d[inum, :-1]  # 使用 inum 索引获取相应的行，同时去掉最后一个元素

        out = d[-1, :-1] / np.minimum(-1e-100, row)  # 用非常小的正数避免除以零

        np.place(out, row >= 0, np.inf)  # 替换无效的比率为无穷大

        if all(x >= 0 for x in row):
            # print('row',row)
            # print(11111111111111111111)
            return None  # 问题无解
        # print(out)

        jnum = np.argmin(out)  # 转入下标
        # print('转入下标',jnum)

        s[inum] = jnum  # 更新基变量

        # print('新基变量',s)
        pivot = d[inum, jnum]
        d[inum] /= pivot
        for i in range(d.shape[0]):
            if i != inum:
                d[i] -= d[i][jnum] * d[inum]
    # print(d)
    num_decision_vars = c.size
    x_values = np.zeros(num_decision_vars)

    for index, value in zip(s, d[:, -1]):
        if index < num_decision_vars:  # 确保索引在非松弛变量的范围内
            x_values[index] = value

    # print([float(value) for value in x_values])
    # 获取最后一行并排除最后一个元素
    last_row_excluding_last = d[-1, :-1]
    # print([float(value) for value in last_row_excluding_last])
    # print(np.any(last_row_excluding_last < 0 - 1e-10))
    # 检查所有元素是否大于0
    if np.any(last_row_excluding_last < 0):
        # print('not')
        solution = simplex_method_high_precision(d, s, c)
        # print([float(value) for value in solution[0]])
        if solution is None:
            return None
        return solution[0], solution[2], solution[1], solution[3]
    else:
        # print(2)
        return x_values, np.sum(c * [float(value) for value in x_values]).astype(float), s, d


def simplex_method(c, A, b, S, B, E, lb, ub, solution = None):
    # print('开始')
    # 设置容差
    tol = 1e-8
    if solution is None:
        solution = simplex_method_no_bound(c, A, b, S, B, E)

    # print(solution)
    if solution is None:
        return None
    else:
        all_satisfied = False
        # 遍历不满足上下界的元素
        while not all_satisfied:
            all_satisfied = True  # 假设所有元素都满足条件
            for i, x in enumerate(solution[0]):
                # print(i,x)
                tableau = solution[3]
                s = solution[2]


                if x < lb[i] - tol:
                    # x小于等于下界 更新下界约束
                    # print('初始矩阵',tableau)

                    # 创造全零行
                    new_row = np.zeros(tableau.shape[1])
                    # i处为-1
                    new_row[i] = -1
                    tableau = np.insert(tableau, -1, new_row, axis=0)  # 将新行插入到原数组的末尾

                    # 在倒数第二列的位置插入新列
                    tableau = np.insert(tableau, -1, 0, axis=1)

                    # 插入松弛变量
                    tableau[-2, -2] = 1

                    # 插入rhs
                    tableau[-2, -1] = -lb[i]
                    # print('插入矩阵',tableau)

                    s.append(tableau.shape[1] - 2)
                    # print(s)

                    tableau = elementary_row_operations(tableau, s)

                    # print('变换矩阵',tableau)
                    # print(11111111111)

                    solution = dual_simplex_method(tableau, s, c)

                    if solution is not None:
                        tableau = solution[3]
                        s = solution[2]
                        all_satisfied = False  # 如果有元素不满足条件，则设置为False，需要重新检查所有元素
                    else:
                        return None

                if x > ub[i] + tol:
                    # print('num',i)
                    # print('v：',x,'ub:',ub[i])
                    # print(tableau)
                    # x大于等于上界 更新上界约束

                    # x小于等于下界 更新下界约束
                    # print('初始矩阵',tableau)

                    # 创造全零行
                    new_row = np.zeros(tableau.shape[1])
                    # i处为-1
                    new_row[i] = 1
                    tableau = np.insert(tableau, -1, new_row, axis=0)  # 将新行插入到原数组的末尾

                    # 在倒数第二列的位置插入新列
                    tableau = np.insert(tableau, -1, 0, axis=1)

                    # 插入松弛变量
                    tableau[-2, -2] = 1

                    # 插入rhs
                    tableau[-2, -1] = ub[i]
                    # print('插入矩阵',tableau)

                    s.append(tableau.shape[1] - 2)
                    # print(s)

                    tableau = elementary_row_operations(tableau, s)

                    # print('变换矩阵',tableau[-2])
                    # print(11111111111)

                    solution = dual_simplex_method(tableau, s, c)

                    if solution is not None:
                        tableau = solution[3]
                        s = solution[2]
                        all_satisfied = False  # 如果有元素不满足条件，则设置为False，需要重新检查所有元素
                    else:
                        return None
        # 如果所有约束条件都满足，则返回当前解
        # print(solution)
        return solution[0], solution[1], solution[2], solution[3]



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


    x = simplex_method(c, A, b, S, B, E, lower_bounds, upper_bounds)
    print(x[1])

