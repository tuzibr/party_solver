from party_solver.optimize import base

model = base.Model()

model.set_linear_params(True)

x = {i: model.add_variable(name=f'x_{i}', vtype='integer', lb=0, ub=200) for i in range(3)}
y = {i: model.add_variable(name=f'y_{i}', vtype='binary', lb=0, ub=1) for i in range(3)}

model.add_constraint(x[0]<=200*y[0])
model.add_constraint(x[1]<=200*y[1])
model.add_constraint(x[2]<=200*y[2])
model.add_constraint(x[0]>=80*y[0])
model.add_constraint(x[1]>=80*y[1])
model.add_constraint(x[2]>=80*y[2])
model.add_constraint(1.5*x[0] + 3*x[1] + 5*x[2] <= 600)
model.add_constraint(280*x[0] + 250*x[1] + 400*x[2] <= 60000)

model.set_objective(2*x[0] + 3*x[1] + 4*x[2],'max')

model.optimize()

model.print_model()
# 输出模型信息、目标函数值和变量值
print('最优函数值:',model.objval())
print('最优解',model.getvars())
