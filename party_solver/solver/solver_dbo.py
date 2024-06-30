import numpy as np
import math

def bound_check(s, Lb, Ub):
    temp = s
    for i in range(len(s)):
        if temp[i] < Lb[i]:
            temp[i] = Lb[i]
        elif temp[i] > Ub[i]:
            temp[i] = Ub[i]
    return temp

def dung_beetle_optimization(objective_function, bounds, num_dung_beetles=30, num_iterations=100):
    """
    使用蜣螂优化算法优化目标函数。

    参数:
    objective_function -- 目标函数，用于评估解决方案的质量。
    bounds -- 优化变量的边界限制，列表中的每个元素都是一个二元组，表示变量的最小和最大值。
    num_dung_beetles -- 参与优化的蜣螂数量，默认为30。
    num_iterations -- 优化算法的迭代次数，默认为100。

    返回:
    best_solution -- 最优解决方案。
    min_value -- 最优解决方案的目标函数值。
    """
    # 初始化下界和上界
    lower_bounds = np.array([bound[0] for bound in bounds])
    upper_bounds = np.array([bound[1] for bound in bounds])
    dimension = len(bounds)

    # 定义种群中蜣螂比例
    prey_percent = 0.2
    num_prey = round(num_dung_beetles * prey_percent)

    scout_percent = 0.2
    num_scout = round(num_dung_beetles * scout_percent)

    wanderer_percent = 0.2
    num_wanderer = round(num_dung_beetles * wanderer_percent)

    # 初始化蜣螂种群的位置和适应度值
    positions = np.random.uniform(lower_bounds, upper_bounds, size=(num_dung_beetles, dimension))
    fitness = np.array([objective_function(*x) for x in positions])

    # 复制种群和适应度值，用于后续更新
    previous_fitness = fitness.copy()
    previous_positions = positions.copy()
    previous_positions_for_wanderers = previous_positions.copy()

    # 找到初始的最佳解决方案
    min_fitness = np.min(fitness)
    best_solution = positions[np.argmin(fitness)]

    # 迭代优化过程
    for iteration in range(num_iterations):
        # 更新蜣螂种群
        most_fit_index = np.argmax(previous_fitness)
        worst_position = previous_positions[most_fit_index]

        for i in range(num_prey):
            # 根据概率选择行为
            if np.random.rand() < 0.9:
                adjustment_factor = 1 if np.random.rand() > 0.1 else -1
                positions[i] = previous_positions[i] + 0.3 * np.abs(previous_positions[i] - worst_position) + \
                               adjustment_factor * 0.1 * (previous_positions_for_wanderers[i])
            else:
                random_angle = np.random.randint(180)
                if random_angle == 0 or random_angle == 90 or random_angle == 180:
                    positions[i] = previous_positions[i]
                theta = random_angle * math.pi / 180
                positions[i] = previous_positions[i] + math.tan(theta) * np.abs(previous_positions[i] - \
                                                                                  previous_positions_for_wanderers[i])

            # 确保解在可行域内
            positions[i] = bound_check(positions[i], lower_bounds, upper_bounds)
            fitness[i] = objective_function(*positions[i])

        # 更新最佳解决方案
        best_position = positions[np.argmin(fitness)]

        # 根据迭代进度调整种群
        reduction_factor = 1 - iteration / num_iterations
        new_position1 = best_position * (1 - reduction_factor)
        new_position2 = best_position * (1 + reduction_factor)
        new_position1 = bound_check(new_position1, lower_bounds, upper_bounds)
        new_position2 = bound_check(new_position2, lower_bounds, upper_bounds)
        new_position11 = best_solution * (1 - reduction_factor)
        new_position22 = best_solution * (1 + reduction_factor)
        new_position11 = bound_check(new_position11, lower_bounds, upper_bounds)
        new_position22 = bound_check(new_position22, lower_bounds, upper_bounds)
        new_lower_bound = new_position1.flatten()
        new_upper_bound = new_position2.flatten()

        # 更新具有繁殖行为的蜣螂
        for i in range(num_prey, num_prey + num_scout):
            positions[i] = best_position + np.random.rand(dimension) * (previous_positions[i] - new_position1) + \
                            np.random.rand(dimension) * (previous_positions[i] - new_position2)
            positions[i] = bound_check(positions[i], new_lower_bound, new_upper_bound)
            fitness[i] = objective_function(*positions[i])

        # 更新其余蜣螂
        for i in range(num_prey + num_scout, num_prey + num_scout + num_wanderer):
            positions[i] = previous_positions[i] + (np.random.randn(dimension)) * \
                            (previous_positions[i] - new_position11) + \
                            (np.random.rand(dimension)) * (previous_positions[i] - new_position22)
            positions[i] = bound_check(positions[i], lower_bounds, upper_bounds)
            fitness[i] = objective_function(*positions[i])

        for j in range(num_prey + num_scout + num_wanderer, num_dung_beetles):
            positions[j] = best_solution + np.random.randn(dimension) * \
                            (np.abs(previous_positions[j] - best_position) + np.abs(previous_positions[j] -
                                                                                   best_solution)) / 2
            positions[j] = bound_check(positions[j], lower_bounds, upper_bounds)
            fitness[j] = objective_function(*positions[j])

        # 更新种群和适应度值
        previous_positions_for_wanderers = previous_positions.copy()
        for i in range(num_dung_beetles):
            if fitness[i] < previous_fitness[i]:
                previous_fitness[i] = fitness[i]
                previous_positions[i] = positions[i]
            if previous_fitness[i] < min_fitness:
                min_fitness = previous_fitness[i]
                best_solution = previous_positions[i]

    return best_solution, min_fitness

if __name__ == '__main__':

    from sympy import lambdify, Symbol
    x = Symbol('x')
    y = Symbol('y')
    func = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    objective_function = lambdify((x, y), func, modules='numpy')

    bounds = [(-10, 10), (-10, 10)]
    num_dung_beetles = 30
    num_iterations = 100

    best_solution, best_fitness = dung_beetle_optimization(objective_function, bounds, num_dung_beetles, num_iterations)
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
