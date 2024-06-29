import numpy as np
import heapq
from party_solver.solver.solver_lp import simplex_method_no_bound
import copy

def is_integer(val, tol=1e-8):
    return abs(val - round(val)) < tol

def round_b_for_integer_constraints(A, b, integer_indices):

    integer_indices = set(integer_indices)  # 转换为集合以加速查找

    for i in range(A.shape[0]):
        non_zero_indices = np.nonzero(A[i])[0]  # 查找非零元素的索引
        if all(idx in integer_indices for idx in non_zero_indices):
            b[i] = np.floor(b[i])  # 向下取整

    return b

def branch_and_bound(c, A, b, S, B, E, lb, ub, integer_indices, compare_objective=float('inf'), log=False):
    def add_branch(problem_queue, counter, bound_type, var_index, bound_value, c, A, b, S, B, E,cut=0):
        # print(bound_type)
        c = copy.deepcopy(c)
        A = copy.deepcopy(A)
        b = copy.deepcopy(b)
        S = copy.deepcopy(S)
        B = copy.deepcopy(B)
        E = copy.deepcopy(E)

        if bound_type == 'lb':

            # 更新 A 矩阵
            new_row = np.zeros(len(c))
            new_row[var_index] = 1
            A_down = np.vstack([A, new_row])

            # 更新 b 向量
            b_down = np.append(b, bound_value)

            # 更新 S 矩阵
            S_down = np.hstack([S, np.zeros((S.shape[0], 1))])  # 增加一列
            S_down = np.vstack([S_down, np.zeros(S_down.shape[1])])  # 增加一行
            S_down[-1, -1] = 1  # 最后一行最后一列设为 1

            # 更新 B 矩阵
            B_down = np.vstack([B, np.zeros(B.shape[1])])  # 增加一行，不立即赋值

            # 更新 E 向量
            E_down = np.append(E, 1)  # 增加一个值为 1，表示小于等于

            solution = simplex_method_no_bound(c, A_down, b_down, S_down, B_down, E_down)
            c, A, b, S, B, E = c, A_down, b_down, S_down, B_down, E_down

        else:  # ub

            # 更新 A 矩阵
            new_row_A = np.zeros(len(c))
            new_row_A[var_index] = 1
            A_up = np.vstack([A, new_row_A])

            # 更新 b 向量
            b_up = np.append(b, bound_value)

            # 更新 S 矩阵
            S_up = np.hstack([S, np.zeros((S.shape[0], 1))])  # 增加一列
            S_up = np.vstack([S_up, np.zeros(S_up.shape[1])])  # 增加一行
            S_up[-1, -1] = -1  # 最后一行最后一列设为 -1

            # 更新 B 矩阵
            B_up = np.hstack([B, np.zeros((B.shape[0], 1))])  # 增加一列
            B_up = np.vstack([B_up, np.zeros(B_up.shape[1])])  # 增加一行
            B_up[-1, -1] = 1  # 最后一行最后一列设为 1

            # 更新 E 向量
            E_up = np.append(E, -1)  # 增加一个值为 -1

            solution = simplex_method_no_bound(c, A_up, b_up, S_up, B_up, E_up)
            c, A, b, S, B, E = c, A_up, b_up, S_up, B_up, E_up

        if solution is not None:
            v = solution[0]
            tol = 1e-8
            # print(solution[1])
            if all(lb[i] - tol <= v[i] <= ub[i] + tol for i in range(len(v))):
                if solution[1] < best_objective:
                    # print("Adding branch with solution:", solution[1])
                    if bound_type == 'lb':
                        heapq.heappush(problem_queue, (solution[1], counter, solution, c, A_down, b_down, S_down, B_down, E_down))

                    else:
                        heapq.heappush(problem_queue, (solution[1], counter, solution, c, A_up, b_up, S_up, B_up, E_up))
                else:
                    if log == True:
                        print("bound cut")
            else:

                all_satisfied = False
                # 遍历不满足上下界的元素
                while not all_satisfied:
                    all_satisfied = True  # 假设所有元素都满足条件

                    for i, x in enumerate(solution[0]):

                        if x < lb[i] - tol:
                            # print('not satisfied', i, x)
                            # 更新 A 矩阵
                            new_row_A = np.zeros(len(c))
                            new_row_A[i] = 1
                            A = np.vstack([A, new_row_A])

                            # 更新 b 向量
                            b = np.append(b, lb[i])

                            # 更新 S 矩阵
                            S = np.hstack([S, np.zeros((S.shape[0], 1))])  # 增加一列
                            S = np.vstack([S, np.zeros(S.shape[1])])  # 增加一行
                            S[-1, -1] = -1  # 最后一行最后一列设为 -1

                            # 更新 B 矩阵
                            B = np.hstack([B, np.zeros((B.shape[0], 1))])  # 增加一列
                            B = np.vstack([B, np.zeros(B.shape[1])])  # 增加一行
                            B[-1, -1] = 1  # 最后一行最后一列设为 1

                            # 更新 E 向量
                            E = np.append(E, -1)  # 增加一个值为 -1

                            solution = simplex_method_no_bound(c, A, b, S, B, E)

                            if solution is not None:

                                all_satisfied = False  # 如果有元素不满足条件，则设置为False，需要重新检查所有元素
                            else:
                                return None

                        if x > ub[i] + tol:
                            # print('not satisfied', i, x)
                            # 更新 A 矩阵
                            new_row = np.zeros(len(c))
                            new_row[i] = 1
                            A = np.vstack([A, new_row])

                            # 更新 b 向量
                            b = np.append(b, ub[i])

                            # 更新 S 矩阵
                            S = np.hstack([S, np.zeros((S.shape[0], 1))])  # 增加一列
                            S = np.vstack([S, np.zeros(S.shape[1])])  # 增加一行
                            S[-1, -1] = 1  # 最后一行最后一列设为 1

                            # 更新 B 矩阵
                            B = np.vstack([B, np.zeros(B.shape[1])])  # 增加一行，不立即赋值

                            # 更新 E 向量
                            E = np.append(E, 1)  # 增加一个值为 1，表示小于等于

                            solution = simplex_method_no_bound(c, A, b, S, B, E)

                            if solution is not None:

                                all_satisfied = False  # 如果有元素不满足条件，则设置为False，需要重新检查所有元素
                            else:
                                return None

                # print("Adding branch with solution:", solution)
                if solution is not None:
                    heapq.heappush(problem_queue, (solution[1], counter, solution, c, A, b, S, B, E))
        else:

            if log == True:
                print('cut index:',var_index)
                print('cut value:',bound_value)


    # 圆整
    b = round_b_for_integer_constraints(A, b, integer_indices)
    initial_solution = simplex_method_no_bound(c, A, b, S, B, E)
    if initial_solution is None:
        return None


    best_solution = None
    global best_objective
    best_objective = float('inf')
    tol = 1e-8
    cut = 0

    initial_objective = initial_solution[1]
    if initial_objective > compare_objective:
        # print(1111111111111)
        return None
    problem_queue = []
    counter = 0
    heapq.heappush(problem_queue, (initial_objective, counter, initial_solution, c, A, b, S, B, E))

    iteration = 0
    while problem_queue:
        iteration += 1
        current_objective, _, solution, c, A, b, S, B, E = heapq.heappop(problem_queue)

        # print(f"Iteration {iteration} - Popped from queue - Current solution:", [float(value) for value in solution[0]])
        # print(f"Iteration {iteration} - Popped from queue - Current solution structure:", type(solution))
        # print(f"Iteration {iteration} - Popped from queue - Current objective:", current_objective)

        x = solution[0]

        all_integer = True

        for i in integer_indices:
            if not is_integer(x[i], tol):
                all_integer = False
                break

        if all_integer:
            if current_objective < best_objective:
                best_solution = x
                best_objective = current_objective
        else:
            var_index = i
            lower_bound = np.floor(x[var_index])
            upper_bound = np.ceil(x[var_index])
            if log == True:
                print('branch:',i)
                print('ub:',lower_bound)
                print('lb:',upper_bound)
            counter += 1
            add_branch(problem_queue, counter, 'lb', var_index, lower_bound, c, A, b, S, B, E,cut)

            counter += 1
            add_branch(problem_queue, counter, 'ub', var_index, upper_bound, c, A, b, S, B, E,cut)

        # print(problem_queue)
        # print(f"Iteration {iteration} - solution so far:", solution[0],solution[1])
        if log == True:
            print(f"Iteration {iteration} - Best objective so far:", best_objective)

    return best_solution, best_objective, solution[2], solution[3], c, A, b, S, B, E

if __name__ == '__main__':
    # Example usage
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
    integer_indices = [0,1,2]  # Indices of variables that need to be integers

    x = branch_and_bound(c, A, b, S, B, E, lower_bounds, upper_bounds, integer_indices)
    print('Best solution:', [float(frac) for frac in x[0]])
