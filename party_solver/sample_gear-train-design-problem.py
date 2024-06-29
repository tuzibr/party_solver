from party_solver.optimize import base
model = base.Model()

x1=model.add_variable(name='x1',lb=12,ub=60)
x2=model.add_variable(name='x2',lb=12,ub=60)
x3=model.add_variable(name='x3',lb=12,ub=60)
x4=model.add_variable(name='x4',lb=12,ub=60)

model.set_objective((1/6.931-(x2*x3/(x1*x4)))**2)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())