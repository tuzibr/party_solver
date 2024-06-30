from party_solver.optimize import base
import sympy as sp

model = base.Model()

model.set_params(sub_method='nm', tol=1e-5)
model.set_params(method='ip')

l2 = model.add_variable(name='l2', lb=0,ub=5,initial_value=3.5)
l3 = model.add_variable(name='l3', lb=0,ub=5,initial_value=2.5)

phi_0 = sp.acos(sp.Min(sp.Max(((1 + l2) ** 2 - l3 ** 2 + 25) / (10 * (1 + l2)), -1), 1))
psi_0 = sp.acos(sp.Min(sp.Max(((1 + l2) ** 2 - l3 ** 2 - 25) / (10 * l3), -1), 1))
T = (90 / 30) * (sp.pi / 180)

fx = 0.0
for i in range(0,30):
    phi_i = phi_0 + i * T
    rho_i = sp.sqrt(26 - 10 * sp.cos(phi_i))
    alpha_i = sp.acos(sp.Min(sp.Max((rho_i ** 2 + l3 ** 2 - l2 ** 2) / (2 * rho_i * l3), -1), 1))
    beta_i = sp.acos(sp.Min(sp.Max((rho_i ** 2 + 24) / (10 * rho_i), -1), 1))

    psi_i = sp.Piecewise((sp.pi - alpha_i - beta_i, (0 <= phi_i) & (phi_i < sp.pi)),
                         (sp.pi - alpha_i + beta_i, True))

    psi_Ei = psi_0 + (2 / (3 * sp.pi)) * (phi_i - phi_0) ** 2
    fx += (psi_i - psi_Ei) ** 2

model.set_objective(fx)

model.add_constraint(1 - l2 <= 0)
model.add_constraint(1 - l3 <= 0)
model.add_constraint(l2 - l3 - 4 <= 0)
model.add_constraint(l3 - l2 - 4 <= 0)
model.add_constraint(6 - l2 - l3 <= 0)
model.add_constraint(l2 ** 2 + l3 ** 2 - 1.414 * l2 * l3 - 16 <= 0)
model.add_constraint(36 - l2 ** 2 - l3 ** 2 - 1.414 * l2 * l3 <= 0)

model.optimize()

model.print_model()
# 输出模型信息、目标函数值和变量值
print('最优函数值:',model.objval())
print('最优解',model.getvars())
