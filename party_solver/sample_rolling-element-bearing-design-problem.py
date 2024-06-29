from party_solver.optimize import base
from sympy import pi,acos,asin,Piecewise
model = base.Model()

model.set_params(sub_method='dbo')
model.set_inspired_params(50,100)

D=160
d=90
Bw=30

Dm=model.add_variable(name='Dm',lb=0.5*(D+d),ub=0.6*(D+d))
Db=model.add_variable(name='Db',lb=0.15*(D-d),ub=0.45*(D-d))
Z=model.add_variable(name='Z',lb=4,ub=50)
fi=model.add_variable(name='fi',lb=0.515,ub=0.6)
f0=model.add_variable(name='f0',lb=0.515,ub=0.6)
KDmin=model.add_variable(name='KDmin',lb=0.4,ub=0.5)
KDmax=model.add_variable(name='KDmax',lb=0.6,ub=0.7)
ep=model.add_variable(name='ep',lb=0.3,ub=0.4)
ee=model.add_variable(name='ee',lb=0.02,ub=0.1)
xi=model.add_variable(name='xi',lb=0.6,ub=0.85)

T=D-d-2*Db
phio=2*pi-acos(((((D-d)/2)-3*(T/4))**2+(D/2-T/4-Db)**2-(d/2+T/4)**2)/(2*((D-d)/2-3*(T/4))*(D/2-T/4-Db)))

model.add_constraint(1+phio/(2*asin(Db/Dm))-Z<=0)
model.add_constraint(-2*Db+KDmin*(D-d)<=0)
model.add_constraint(-KDmax*(D-d)+2*Db<=0)
model.add_constraint(xi*Bw-Db<=0)
model.add_constraint(-Dm+0.5*(D+d)<=0)
model.add_constraint(-(0.5+ee)*(D+d)+Dm<=0)
model.add_constraint(-0.5*(D-Dm-Db)+ep*Db<=0)
model.add_constraint(0.515-fi<=0)
model.add_constraint(0.515-f0<=0)

gama=Db/Dm
fc=37.91*((1+(1.04*((1-gama/1+gama)**1.72)*((fi*(2*f0-1)/f0*(2*fi-1))**0.41))**(10/3))**-0.3)*((gama**0.3*(1-gama)**1.39)/(1+gama)**(1/3))*(2*fi/(2*fi-1))**0.41

f = Piecewise((fc * Z**(2/3) * Db**1.8, Db <= 25.4),
    (3.647 * fc * Z**(2/3) * Db**1.4, True))

model.set_objective(f,'max')

model.set_initial_values([125.0, 21.87500000000001,50.0,  0.515, 0.515, 0.5,  0.677735345670893, 0.3, 0.02000448247287265, 0.7116892234687728])

model.optimize(1)

model.print_model()

print(model.objval())

print(model.getvars())

model.test_function([131.19998000009525,18.000000000000007,27.0, 0.5575,0.5575, 0.45,0.6499999999999999,0.3, 0.060000000000000005,0.6000000000000029]
)


