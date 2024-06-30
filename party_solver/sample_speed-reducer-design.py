from party_solver.optimize import base
model = base.Model()

x1=model.add_variable(name='x1',lb=2.6,ub=3.6)
x2=model.add_variable(name='x2',lb=0.7,ub=0.8)
x3=model.add_variable(name='x3',lb=17,ub=28)
x4=model.add_variable(name='x4',lb=7.3,ub=8.3)
x5=model.add_variable(name='x5',lb=7.3,ub=8.3)
x6=model.add_variable(name='x6',lb=2.9,ub=3.9)
x7=model.add_variable(name='x7',lb=5,ub=5.5)

model.set_objective(0.7854*x1*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.508*x1*(x6**2 + x7**2) + 7.4777*(x6**3+x7**3) + 0.7854*(x4*x6**2 + x5*x7**2))

model.add_constraint((27/(x1*x2**2*x3))-1<=0)
model.add_constraint((397.5/(x1*x2**2*x3**2)) - 1<=0)
model.add_constraint( (1.93*x4**3/(x2*x3*x6**4)) - 1<=0)
model.add_constraint((1.93*x5**3/(x2*x3*x7**4)) - 1<=0)
model.add_constraint((1/(110*x6**3))*(((745*x4/(x2*x3))**2 + 16.9*1e6)**0.5)-1<=0)
model.add_constraint((1/(85*x7**3))*(((745*x5/(x2*x3))**2 + 157.5*1e6)**0.5)-1<=0)
model.add_constraint( (x2*x3/40) - 1<=0)
model.add_constraint((5*x2/x1) - 1<=0)
model.add_constraint((x1/(12*x2)) - 1<=0)
model.add_constraint( ((1.5*x6+1.9)/(x4)) - 1<=0)
model.add_constraint(((1.1*x7+1.9)/(x5)) - 1<=0)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())