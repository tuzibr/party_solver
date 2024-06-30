from party_solver.optimize import base
model = base.Model()

x1=model.add_variable(name='d',lb=0.05,ub=2)
x2=model.add_variable(name='D',lb=0.25,ub=1.3)
x3=model.add_variable(name='N',lb=2,ub=15)


model.set_objective((x3+2)*x2*(x1**2))

model.add_constraint(1-((x2**3)*x3)/(71785*(x1**4))<=0)
model.add_constraint((4*(x2**2)-x1*x2)/(12566*(x2*(x1**3)-(x1**4))) + 1/(5108*(x1**2))-1<=0)
model.add_constraint(1-(140.45*x1)/((x2**2)*x3)<=0)
model.add_constraint(((x1+x2)/1.5)-1<=0)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())