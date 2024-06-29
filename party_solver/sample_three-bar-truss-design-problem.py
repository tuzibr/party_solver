from party_solver.optimize import base
from sympy import sqrt
model = base.Model()

x1=model.add_variable(name='x1',lb=0,ub=1)
x2=model.add_variable(name='x2',lb=0,ub=1)

l = 100
P = 2
q = 2

model.set_objective(l*(2*sqrt(2)*x1+x2))

model.add_constraint(P*(sqrt(2)*x1+x2)/(sqrt(2)*x1**2+2*x1*x2)-q<=0)
model.add_constraint(P*(x2)/(sqrt(2)*x1**2+2*x1*x2)-q<=0)
model.add_constraint(P/(sqrt(2)*x2+x1)-q<=0)


model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())