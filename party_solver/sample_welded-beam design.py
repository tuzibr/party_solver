from party_solver.optimize import base

model = base.Model()

model.set_linear_params(False)

model.set_params(num_particles=50, max_iter_pso=500)
model.set_params(sub_method='pso', tol=1e-8)

# 添加优化变量，限定变量的上下界
x1 = model.add_variable(name='h', lb=0.1, ub=2, initial_value=0.5)
x2 = model.add_variable(name='l', lb=0.1, ub=10, initial_value=1)
x3 = model.add_variable(name='t', lb=0.1, ub=10, initial_value=1)
x4 = model.add_variable(name='b', lb=0.1, ub=2, initial_value=1)

# 定义常量
L = 14
E = 30 * 10**6
P = 6000
G = 12 * 10**6

# 定义中间变量和公式
M = P * (L + x2/2)
R = ((x2**2 + (x1 + x3)**2)/ 4)**0.5
J = 2 * ((2**0.5 * x1 * x2) * (x2**2 / 4 + (x1 + x3)**2 / 4))

tau_prime = P / (2**0.5 * x1 * x2)
tau_double_prime = M * R / J
tau = (tau_prime**2 + (2 * tau_prime * tau_double_prime * x2)/ (2 * R) + tau_double_prime**2 )**0.5

sigma = 6*P*L / (x4 * x3**2)
delta = 6*P*L**3 / (E*x4 * x3**2)
Pc = (4.013 * E * (1 - (E/(4*G))**0.5 * x3/(2*L)) * x3 * x4**3) / (6 * L**2)

# 设置目标函数
model.set_objective((2.20942*x1**2*0.5*x2 + 0.04811*x3*x4*(L+x2)))

# 添加约束条件
model.add_constraint(tau <= 13600)
model.add_constraint(sigma <= 30000)
model.add_constraint(delta <= 0.25)
model.add_constraint(Pc >= 6000)
model.add_constraint(x1 <= x4)
model.add_constraint(x1 >= 0.125)

model.optimize()
model.print_model()
# 输出模型信息、目标函数值和变量值
print('最优函数值:',model.objval())
print('最优解',model.getvars())

