from party_solver.optimize import base
from sympy import pi
model = base.Model()

x1=model.add_variable(name='Ts',lb=0,ub=99)
x2=model.add_variable(name='Th',lb=0,ub=99)
x3=model.add_variable(name='R',lb=10,ub=200)
x4=model.add_variable(name='L',lb=10,ub=200)

model.set_objective(0.6224*x1*x3*x4 + 1.7781*x2*x3**2+3.1661*x1**2*x4+19.84*x1**2*x3)

model.add_constraint(-x1+0.0193*x3<=0)
model.add_constraint(-x2+0.00954*x3<=0)
model.add_constraint(-pi*x3**2*x4 - (4/3)*pi*x3**3 + 1296000<=0)
model.add_constraint(x4 - 240<=0)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())