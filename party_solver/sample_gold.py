from party_solver.optimize import base
from sympy import cos,sin
model = base.Model()

model.set_params(method='gold')

x = model.add_variable(name="x", lb=0, ub=6.28)

# model.set_objective(cos(x)+sin(x))
model.set_objective(x**2-3*x+2)

model.optimize()



