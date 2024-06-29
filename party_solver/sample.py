from party_solver.optimize import base
from sympy import cos, pi
model = base.Model()

model.set_params(method='ip')

# model.set_params(num_particles=50, max_iter_pso=200)
x = {i:model.add_variable(name=f'x_{i}',lb=1,ub=5.12,initial_value=5)for i in range(2)}

# rastrigin_expr = [10 + x[i]**2 - 10 * cos(2 * pi * x[i]) for i in range(10)]
# model.set_objective(model.quicksum(rastrigin_expr))

model.set_objective(model.quicksum(x[i]**2 for i in range(2)))
# model.add_constraint(model.Eq(model.quicksum(x[i]**2 for i in range(2)), 1))
model.add_constraint(model.quicksum(x[i]**2 for i in range(2))>=1)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())


