from party_solver.optimize import base
model = base.Model()

x={i:model.add_variable(name=f'x_{i}',lb=0.01,ub=100) for i in range(1,6)}

model.set_objective(0.0624*(model.quicksum(x[i] for i in range(1,6))))

model.add_constraint(61/x[1]**3+37/x[2]**3+19/x[3]**3+7/x[4]**3+1/x[5]**3-1<=0)

model.optimize()

model.print_model()

print(model.objval())

print(model.getvars())