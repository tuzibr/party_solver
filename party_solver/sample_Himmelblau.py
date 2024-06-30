from party_solver.optimize import base

# 初始化优化模型以及设定求解模式为非线性，包括粒子群优化算法的具体参数
model = base.Model()
model.set_linear_params(False)
model.set_params(num_particles=50, max_iter_pso=200)
model.set_params(method='alm')
model.set_params(sub_method='pso')

x = model.add_variable(name='x',lb=-10, ub=10)
y = model.add_variable(name='y',lb=-10, ub=10)

model.set_objective((x**2 + y - 11)**2+(x+y**2-7)**2, 'min')

test_time = 10
model.optimize(num_runs=test_time)

# 输出模型信息、目标函数值和变量值
print('最优函数值:',model.objval())
print('最优解',model.getvars())
# 提取局部最优解方案
model.extract_local_optima_solutions()



