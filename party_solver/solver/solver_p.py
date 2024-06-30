import numpy as np
import sympy as sp
from party_solver.solver import solver_pso
from party_solver.solver import solver_gd
from party_solver.solver import solver_dn
from party_solver.solver import solver_qn
from party_solver.solver import solver_lbfgsb
from party_solver.solver import solver_cg
from party_solver.solver import solver_pm
from party_solver.solver import solver_nm
from party_solver.solver import solver_dbo
from party_solver.solver import solver_tr
import random

def calmin(objective_function, equality_constraints, inequality_constraints, bounds, vars, x0=[], tol=1e-6,
           max_iter=100,
           sigma=5, method='lbfgsb',
           num_particles=50, max_iter_pso=200, w_max=0.9, w_min=0.4, cognitive_start=2.5, cognitive_end=0.5,
           social_start=0.5, social_end=2.5):

    stay_flag = False

    if len(x0) == 0 :
        x0 = np.array([(x + y) / 2 for x, y in bounds])
    else:
        x0 = np.array(x0)

    for iteration in range(max_iter):
        if iteration == max_iter - 1:
            raise ValueError("Optimal Failed, Check Constraint")

        if stay_flag:
            x0 = np.array([random.uniform(x, y) for x, y in bounds])
            stay_flag = False

        # 惩罚函数
        punished_function = objective_function \
                               + sigma * sum(c ** 2 for c in equality_constraints) \
                               + sigma * sum(sp.Max(0, ic) ** 2 for ic in inequality_constraints)

        punished_function_callable = sp.lambdify(vars, punished_function)
        try:
            if method == 'gd':
                result_vars = solver_gd.gradient_descent(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'dn':
                result_vars = solver_dn.damped_newton_optimization(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'qn':
                result_vars = solver_qn.quasi_newton_optimization(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'lbfgsb':
                result_vars = solver_lbfgsb.l_bfgs_b_optimization(punished_function, vars, x0, epsilon=tol,bounds=bounds)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'cg':
                result_vars = solver_cg.conjugate_gradient_optimization(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'pm':
                result_vars = solver_pm.powell_optimization(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'nm':
                result_vars = solver_nm.nelder_mead_optimization_sympy(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'tr':
                result_vars = solver_tr.trust_region_optimization(punished_function, vars, x0, epsilon=tol)
                result_obj = punished_function_callable(*result_vars)
                # print(result_obj)
                # print(result_vars)
                if result_obj == -np.inf:
                    raise Exception
            elif method == 'dbo':
                result_vars, result_obj = solver_dbo.dung_beetle_optimization(punished_function_callable,
                                                                              bounds,
                                                                              num_dung_beetles=num_particles,
                                                                              num_iterations=max_iter_pso)
            elif method == 'pso':
                result_vars, result_obj, _ = solver_pso.particle_swarm_optimization(punished_function_callable,
                                                                                    bounds,
                                                                                    num_particles=num_particles,
                                                                                    max_iter=max_iter_pso,
                                                                                    w_max=w_max,
                                                                                    w_min=w_min,
                                                                                    cognitive_start=cognitive_start,
                                                                                    cognitive_end=cognitive_end,
                                                                                    social_start=social_start,
                                                                                    social_end=social_end)
            else:
                raise ValueError(f"Method {method} not found. Use 'gd', 'qn', 'lbfgsb', 'cg', 'pm', 'nm', 'dbo', 'pso'")
        except:
            raise ValueError(f"Method {method} failed, Switching to Other Method.")

        solution_dict = dict(zip(vars, result_vars))

        eq_constraints_values = np.array([float(c.subs(solution_dict)) for c in equality_constraints])
        ineq_constraints_values = np.array([float(ic.subs(solution_dict)) for ic in inequality_constraints])

        eq_satisfied = any(eq_constraints_values)
        ineq_satisfied = any(ineq_constraints_values)

        if eq_satisfied and ineq_satisfied:
            print(
                f'Iteration {iteration + 1}: \nObjective value = {result_obj}, \nResult vars = {result_vars}, \nEquality constraints = {eq_constraints_values}, \nInequality constraints = {ineq_constraints_values}\n')
        else:
            if not eq_satisfied and not ineq_satisfied:
                print(f'Iteration {iteration + 1}: \nObjective value = {result_obj}, \nResult vars = {result_vars}\n')
            elif not eq_satisfied:
                print(
                    f'Iteration {iteration + 1}: \nObjective value = {result_obj}, \nResult vars = {result_vars}, \nInequality constraints = {ineq_constraints_values}\n')
            elif not ineq_satisfied:
                print(
                    f'Iteration {iteration + 1}: \nObjective value = {result_obj}, \nResult vars = {result_vars}, \nEquality constraints = {eq_constraints_values}\n')

        if np.sum(np.abs(eq_constraints_values)) + np.sum(
                np.maximum(0, ineq_constraints_values))< tol * 1e1 :
            print(f'p convergence achieved after {iteration + 1} iterations')
            break

        if np.linalg.norm(np.array(result_vars) - np.array(x0))==0:
            print('switch to random initial vars')
            stay_flag = True

        sigma *= 2
        x0 = result_vars

    return result_vars, result_obj


if __name__ == '__main__':
    x, y = sp.symbols('x y')

    objectfunction = x ** 2 + y ** 2
    eq_constraint1 = x + y - 3
    eq_constraint2 = x - y
    ineq_constraint1 = -x
    ineq_constraint2 = -y
    equality_constraints = [eq_constraint1, eq_constraint2]
    inequality_constraints = [ineq_constraint1, ineq_constraint2]

    # 运行优化
    result = calmin(objectfunction, equality_constraints, inequality_constraints, [(0, 10), (0, 10)], (x, y),
                    method='lbfgsb')
    print(result)
