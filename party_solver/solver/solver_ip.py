import numpy as np
from party_solver.tools import difftools
import sympy as sp
import time


def bisection_method(f, tol=1e-5, max_iter=100):
    a = -tol
    b = 1
    if f(a) >= 0:
        if f(b) >= 0:
            return 1
        else:
            for _ in range(max_iter):
                c = (a + b) / 2
                if f(c) == 0 or (b - a) / 2 < tol:
                    return c
                if f(a) * f(c) < 0:
                    b = c
                else:
                    a = c
            return a
    elif f(a) >= -tol * 100:
        return 0
    else:
        print(f(a))
        raise ValueError


def kkt_system(H, J_h, J_g, S, Lambda, grad_f, y, lambda_, mu, e, h, g, s,
               sigma=1, lambda_aff=None, s_aff=None,
               flag=None):
    # print(H)
    # print(J_h)
    # print(J_g)
    # print(S)
    # print(Lambda)
    # print(grad_f)
    # print(y)
    # print(lambda_)
    # print(lambda_)
    # print(e)
    # print(h)
    # print(g)
    # print(s)
    # print('--------------------------------------------')

    n = len(grad_f)
    p = len(y)
    m = len(s)
    if lambda_aff is None and s_aff is None:
        lambda_aff = Lambda * 0
        s_aff = S * 0
    if p != 0 and m != 0:

        block1 = np.hstack((H, np.zeros((n, m)), J_h.T, J_g.T))
        block2 = np.hstack((np.zeros((m, n)), Lambda, np.zeros((m, p)), S))
        block3 = np.hstack((J_h, np.zeros((p, m)), np.zeros((p, p)), np.zeros((p, m))))
        block4 = np.hstack((J_g, np.eye(m, m), np.zeros((m, p)), np.zeros((m, m))))

        A = np.vstack((block1, block2, block3, block4))

        b = -np.concatenate([
            grad_f + J_h.T @ y + J_g.T @ lambda_,
            S @ lambda_ + lambda_aff @ s_aff @ np.ones(m) - mu * e * sigma,
            h,
            g + s
        ])

    elif p == 0 and m != 0:
        block1 = np.hstack((H, np.zeros((n, m)), J_g.T))
        block2 = np.hstack((np.zeros((m, n)), Lambda, S))
        block3 = np.hstack((J_g, np.eye(m, m), np.zeros((m, m))))

        A = np.vstack((block1, block2, block3))
        # print(lambda_aff@s_aff)
        b = -np.concatenate([
            grad_f + J_g.T @ lambda_,
            S @ lambda_ + lambda_aff @ s_aff @ np.ones(m) - mu * e * sigma,
            (g + s)
        ])
        # print(lambda_)
        # print(A,b)
        # print(np.linalg.solve(A, b))
    if flag is None:
        return np.linalg.solve(A, b)
    elif flag is True:
        return np.linalg.pinv(A), b
    elif flag is False:
        return A, b


# 更新障碍参数
def update_barrier_parameter(mu, pred_step_size, accuracy_threshold=1e-3):
    if pred_step_size < accuracy_threshold:
        return mu * 0.01
    else:
        return mu * 0.2


def judge_function(func, x0, h, g, y, lambda_, s, mu):
    judge = func(*x0) + mu * np.sum(np.log(s)) + np.sum(y * np.linalg.norm(h)) + np.linalg.norm((g + s) * lambda_)
    return judge


# 内点法主算法
def interior_point_method(func, vars, x0, eqcons, ieqcons, bounds,
                          tol=1e-6, max_iter=1000, mu=10, barrierparamupdate=True):
    func_callable = sp.lambdify(vars, func, 'numpy')
    judge_value = np.inf
    if len(eqcons) == 0:
        eq_constraints = [sp.lambdify(vars, con, 'numpy') for con in eqcons]
        ineq_constraints = [sp.lambdify(vars, con, 'numpy') for con in ieqcons]

        x = x0
        s = np.ones(len(ineq_constraints))  # 初始化松弛变量s
        e = np.ones(len(s))
        lambda_ = 2 * np.ones(len(s))
        y = np.ones(len(eq_constraints))
        n = len(ineq_constraints) + len(eq_constraints)
    else:
        eq_constraints = [sp.lambdify(vars, con, 'numpy') for con in eqcons]
        ineq_constraints = [sp.lambdify(vars, con, 'numpy') for con in ieqcons]

        x = x0
        s = np.ones(len(ineq_constraints))  # 初始化松弛变量s
        e = np.ones(len(s))
        lambda_ = 2 * np.ones(len(s))
        y = np.ones(len(eq_constraints))
        n = len(ineq_constraints) + len(eq_constraints)

    if barrierparamupdate:

        grad_f = difftools.gradient(func, vars, x)
        H = difftools.hessian(func, vars, x)
        J_h = difftools.jacobian(eq_constraints, vars, x)
        J_g = difftools.jacobian(ineq_constraints, vars, x)
        S = np.diag(s)
        Lambda = np.diag(lambda_)
        h = [con(*x) for con in eq_constraints]
        g = [con(*x) for con in ineq_constraints]

        dx_s = kkt_system(H, J_h, J_g, S, Lambda, grad_f, y, lambda_, mu, e, h, g, s, 0)
        dx, ds, dy, dlambda = np.split(dx_s, [len(x), len(x) + len(s), len(x) + len(s) + len(y)])

        alpha = 1
        # 更新变量
        s = np.maximum(abs(s + alpha * ds), 1)
        lambda_ = np.maximum(abs(lambda_ + alpha * dlambda), 1)

        # 预测矫正步
        for iteration in range(max_iter):
            # print(x)

            # 计算当前梯度和黑塞矩阵
            grad_f = difftools.gradient(func, vars, x)
            H = difftools.hessian(func, vars, x)
            J_h = difftools.jacobian(eq_constraints, vars, x)
            J_g = difftools.jacobian(ineq_constraints, vars, x)
            S = np.diag(s)
            Lambda = np.diag(lambda_)
            h = [con(*x) for con in eq_constraints]
            g = [con(*x) for con in ineq_constraints]
            mu = np.sum(lambda_ * s) / n

            judge_new = judge_function(func_callable, x, h, g, y, lambda_, s, mu)
            if judge_new < 2 * judge_value:
                judge_value = judge_new
            else:
                print(f'judge_value is not decrease {judge_new, judge_value}')
                # raise ValueError

            packet = kkt_system(H, J_h, J_g, S, Lambda, grad_f, y, lambda_, mu, e, h, g, s, 0, flag=True)
            A_pinv, b = packet[0], packet[1]
            dx_s = np.dot(A_pinv, b)
            dx_aff, ds_aff, dy_aff, dlambda_aff = np.split(dx_s, [len(x), len(x) + len(s), len(x) + len(s) + len(y)])

            alpha_lamda_max = []
            for lambda__, dlambda_ in zip(lambda_, dlambda_aff):
                alpha_lamda_max.append(bisection_method(lambda alpha: lambda__ + alpha * dlambda_))
            alpha_lamda_max = min(alpha_lamda_max)
            alpha_s_max = []
            for s_, ds_ in zip(s, ds_aff):
                alpha_s_max.append(bisection_method(lambda alpha: s_ + alpha * ds_))
            alpha_s_max = min(alpha_s_max)

            alpha_aff = min(alpha_lamda_max, alpha_s_max)

            mu_aff = np.sum((lambda_ + alpha_aff * dlambda_aff) * (s + alpha_aff * ds_aff)) / n

            sigma = (mu_aff / mu) ** 3

            b = kkt_system(H, J_h, J_g, S, Lambda,
                           grad_f, y, lambda_, mu, e, h, g, s,
                           sigma, np.diag(dlambda_aff), np.diag(ds_aff), flag=False)[1]
            dx_s = np.dot(A_pinv, b)
            dx, ds, dy, dlambda = np.split(dx_s, [len(x), len(x) + len(s), len(x) + len(s) + len(y)])

            tau = 1 - 0.5 ** (iteration + 1)

            alpha_pri = []
            for lambda__, dlambda_ in zip(lambda_, dlambda):
                alpha_pri.append(bisection_method(lambda alpha: lambda__ + alpha * dlambda_ - (1 - tau) * lambda__))
            alpha_pri = min(alpha_pri)
            alpha_dual = []
            for s_, ds_ in zip(s, ds):
                alpha_dual.append(bisection_method(lambda alpha: s_ + alpha * ds_ - (1 - tau) * s_))
            alpha_dual = min(alpha_dual)
            alpha = min(alpha_pri, alpha_dual)

            x = x + alpha * dx
            s = s + alpha * ds
            y = y + alpha * dy
            lambda_ = lambda_ + alpha * dlambda

            # 检查收敛
            if np.any(abs(np.block([dx, ds, dy, dlambda])) < tol * np.ones(len(x) + len(s) + len(y) + len(lambda_))):
                print(f'Interior Point Method Converged after {iteration} iterations.')
                break


    else:
        # 直接步
        for iteration in range(max_iter):
            # 计算当前梯度和海塞矩阵
            grad_f = difftools.gradient(func, vars, x)
            H = difftools.hessian(func, vars, x)
            J_h = difftools.jacobian(eq_constraints, vars, x)
            J_g = difftools.jacobian(ineq_constraints, vars, x)
            S = np.diag(s)
            Lambda = np.diag(lambda_)
            h = [con(*x) for con in eq_constraints]
            g = [con(*x) for con in ineq_constraints]

            dx_s = kkt_system(H, J_h, J_g, S, Lambda, grad_f, y, lambda_, mu, e, h, g, s, 1)
            dx, ds, dy, dlambda = np.split(dx_s, [len(x), len(x) + len(s), len(x) + len(s) + len(y)])

            alpha_pri = []
            for lambda__, dlambda_ in zip(lambda_, dlambda):
                alpha_pri.append(bisection_method(lambda alpha: lambda__ + alpha * dlambda_))
            alpha_pri = min(alpha_pri)

            alpha_dual = []
            for s_, ds_ in zip(s, ds):
                alpha_dual.append(bisection_method(lambda alpha: s_ + alpha * ds_))
            alpha_dual = min(alpha_dual)

            alpha = min(alpha_pri, alpha_dual)

            x = x + alpha * dx
            s = s + alpha * ds
            y = y + alpha * dy
            lambda_ = lambda_ + alpha * dlambda

            # 更新障碍参数
            mu = update_barrier_parameter(mu, max(np.linalg.norm(grad_f + J_h.T @ y + J_g.T @ lambda_),
                                                  np.linalg.norm(S @ lambda_ - mu * e),
                                                  np.linalg.norm(h),
                                                  np.linalg.norm(g + s)))

            # 检查收敛
            if np.any(abs(np.block([dx, ds, dy, dlambda])) < tol * np.ones(len(x) + len(s) + len(y) + len(lambda_))):
                print(f'Interior Point Method Converged after {iteration} iterations.')
                break

    return x, func_callable(*x)


if __name__ == '__main__':
    x1, x2 = sp.symbols('x1 x2')

    func = (x1 ** 2 + 2 * x2 ** 2 - 2 * x1 * x2) * 0.5 - 2 * x1 - 6 * x2

    eqcons = []
    ieqcons = [-x1, -x2]

    # 初始值
    x0 = np.array([4, 4])

    bounds = [(1, 3), (1, 3)]

    # 调用内点法
    x = interior_point_method(func, (x1, x2), x0, eqcons, ieqcons, bounds)

    print("Optimal x:", x)
