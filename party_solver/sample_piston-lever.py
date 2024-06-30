from party_solver.optimize import base
from sympy import cos, pi, sin
model = base.Model()

x1=model.add_variable(name='H',lb=0.05,ub=500)
x2=model.add_variable(name='B',lb=0.05,ub=500)
x3=model.add_variable(name='D',lb=0.05,ub=120)
x4=model.add_variable(name='X',lb=0.05,ub=500)

theta = pi/4
Q = 10000
L = 240
M = 1.8*10**6
P = 1500

F=pi*P*x3**2/4
R = abs(-x4*(x4*sin(theta)+x1)+x1*(x2-x4*cos(theta)))/((x4-x2)**2+x1**2)**0.5
L1=((x4-x2)**2+x1**2)**0.5
L2=((x4*sin(theta)+x1)**2+(x2-x4*cos(theta))**2)**0.5

model.set_objective(0.25*pi*x3**2*(L2-L1))

model.add_constraint(Q*L*cos(theta)-R*F<=0)
model.add_constraint(Q*(L-x4)-M<=0)
model.add_constraint(1.2*(L2-L1)-L1<=0)
model.add_constraint(0.5*x3-x2<=0)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())