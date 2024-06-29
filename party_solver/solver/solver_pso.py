import numpy as np
from pyDOE import lhs


def lhs_numpy(dimensions, samples):
    """
    使用NumPy实现拉丁超立方抽样

    参数:
    dimensions - 抽样的维度数
    samples - 需要生成的样本数量

    返回:
    一个形状为(samples, dimensions)的二维数组，包含生成的LHS样本
    """
    # 初始化一个全零的二维数组用于存放样本
    lhs_array = np.zeros((samples, dimensions))

    # 对于每个维度
    for dim in range(dimensions):
        # 生成一个在[0, 1)范围内的均匀分布随机数序列
        random_nums = np.random.rand(samples)
        # 对这些随机数进行排序，但保持原始随机数的相对顺序（这是LHS的关键步骤）
        sorted_indices = np.argsort(random_nums)
        # 生成每个维度的样本点，确保每个子区间内只有一个样本点
        lhs_array[:, dim] = np.array([random_nums[i] + (i / samples) for i in sorted_indices])
        # 确保最后一个样本点不会超过1（由于浮点数运算可能的误差）
        lhs_array[-1, dim] = 1.0

    return lhs_array

def particle_swarm_optimization(func, bounds, num_particles=30, max_iter=300, w_max=0.9, w_min=0.4, cognitive_start=2.5, cognitive_end=0.5, social_start=0.5, social_end=2.5):
    """
    使用粒子群优化算法优化函数。

    参数:
    func: 需要优化的函数。
    bounds: 参数的边界限制，格式为[(min1, max1), (min2, max2), ...]。
    num_particles: 粒子群中粒子的数量。
    max_iter: 迭代的最大次数。
    w_max: 最大惯性权重。
    w_min: 最小惯性权重。
    cognitive_start: 初始认知学习因子。
    cognitive_end: 最终认知学习因子。
    social_start: 初始社会学习因子。
    social_end: 最终社会学习因子。

    返回:
    global_best_position: 全局最优解的位置。
    best_value: 全局最优解的值。
    pso_solutions: 迭代过程中全局最优解的轨迹。
    """
    # 确定参数维度
    dimensions = len(bounds)

    # 计算每个维度的最大速度
    vmax = np.array([0.2 * (bound[1] - bound[0]) for bound in bounds])

    # 使用 LHS 初始化粒子位置
    try:
        lhs_samples = lhs(dimensions, samples=num_particles)
    except:
        lhs_samples = lhs_numpy(dimensions, num_particles)
    particles_position = np.zeros_like(lhs_samples)
    for i in range(dimensions):
        particles_position[:, i] = lhs_samples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    # 随机初始化粒子速度
    particles_velocity = np.random.uniform(-vmax, vmax, (num_particles, dimensions))

    # 初始化每个粒子的个人最佳位置和值
    personal_best_position = np.copy(particles_position)
    personal_best_value = np.array([func(*pos) for pos in personal_best_position])
    # 初始化全局最佳位置
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]

    # 初始化记录全局最优解的列表和没有改进的计数器
    pso_solutions = [global_best_position]
    no_improvement_count = [0] * num_particles
    no_improvement_threshold = 10

    # 主循环：迭代寻找最优解
    for iteration in range(max_iter):
        # 计算当前迭代的惯性权重、认知学习因子和社会学习因子
        inertia = w_max - ((w_max - w_min) * (iteration / max_iter))
        cognitive = cognitive_start - ((cognitive_start - cognitive_end) * (iteration / max_iter))
        social = social_start + ((social_end - social_start) * (iteration / max_iter))

        # 更新每个粒子的速度和位置
        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            particles_velocity[i] = (inertia * particles_velocity[i] +
                                     cognitive * r1 * (personal_best_position[i] - particles_position[i]) +
                                     social * r2 * (global_best_position - particles_position[i]))
            particles_velocity[i] = np.clip(particles_velocity[i], -vmax, vmax)

            particles_position[i] += particles_velocity[i]
            particles_position[i] = np.clip(particles_position[i], [bound[0] for bound in bounds], [bound[1] for bound in bounds])
            # 应用边界反弹策略
            if np.any(particles_position[i] == [bound[0] for bound in bounds]) or np.any(
                    particles_position[i] == [bound[1] for bound in bounds]):
                particles_velocity[i] = -particles_velocity[i]  # 反弹边界策略

            # 更新个人最佳位置和值
            current_value = func(*particles_position[i])
            if current_value < personal_best_value[i]:
                personal_best_position[i] = np.copy(particles_position[i])
                personal_best_value[i] = current_value
                no_improvement_count[i] = 0
            else:
                no_improvement_count[i] += 1
                # 如果一段时间内没有改进，则重新初始化粒子
                if no_improvement_count[i] > no_improvement_threshold:
                    particles_position[i] = np.random.uniform([bound[0] for bound in bounds],
                                                              [bound[1] for bound in bounds], dimensions)
                    particles_velocity[i] = np.random.uniform(-vmax, vmax, dimensions)
                    no_improvement_count[i] = 0

        # 更新全局最佳位置
        global_best_index = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_index]
        pso_solutions.append(global_best_position)

        # 检查收敛条件
        # 收敛判定
        if np.std(personal_best_value) < 1e-6:
            break

    # 返回全局最优解的位置、值和整个优化过程的解决方案轨迹
    best_value = func(*global_best_position)
    return global_best_position, best_value, pso_solutions

if __name__ == '__main__':
    from sympy import lambdify, Symbol
    import matplotlib.pyplot as plt

    x = Symbol('x')
    y = Symbol('y')
    himmelblau_func = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    objective_function = lambdify((x, y), himmelblau_func, modules='numpy')

    bounds = [(-10, 10), (-10, 10)]
    num_dung_beetles = 100
    num_iterations = 100

    best_solution, best_fitness, pso_solutions = particle_swarm_optimization(objective_function, bounds, num_dung_beetles, num_iterations)

    x_vals = np.linspace(-5, 5, 400)
    y_vals = np.linspace(-5, 5, 400)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    z_vals = objective_function(x_grid, y_grid)

    plt.figure(figsize=(12, 6))

    plt.contour(x_grid, y_grid, z_vals, levels=50, cmap='viridis')
    plt.colorbar()

    pso_solutions = np.array(pso_solutions)
    plt.plot(pso_solutions[:, 0], pso_solutions[:, 1], marker='o', color='red')
    plt.scatter(pso_solutions[-1, 0], pso_solutions[-1, 1], marker='x', color='blue', s=100, label='Best Solution')

    plt.title("PSO Optimization Trajectory on Himmelblau's Function")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
