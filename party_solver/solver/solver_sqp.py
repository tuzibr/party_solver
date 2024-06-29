import numpy as np
import warnings
from party_solver.tools import difftools
import sympy as sp
from party_solver.solver.slsqp import slsqp
from numpy import (zeros, array, append, concatenate, finfo,
                   sqrt, vstack, isfinite, atleast_1d)

class OptimizeWarning(UserWarning):
    pass

_epsilon = sqrt(finfo(float).eps)

def _arr_to_scalar(x):
    # If x is a numpy array, return x.item().  This will
    # fail if the array has more than one element.
    return x.item() if isinstance(x, np.ndarray) else x

def old_bound_to_new(bounds):
    lb, ub = zip(*bounds)

    # Convert occurrences of None to -inf or inf, and replace occurrences of
    # any numpy array x with x.item(). Then wrap the results in numpy arrays.
    lb = np.array([float(_arr_to_scalar(x)) if x is not None else -np.inf
                   for x in lb])
    ub = np.array([float(_arr_to_scalar(x)) if x is not None else np.inf
                   for x in ub])

    return lb, ub

def _clip_x_for_func(func, bounds):
    # ensures that x values sent to func are clipped to bounds

    # this is used as a mitigation for gh11403, slsqp/tnc sometimes
    # suggest a move that is outside the limits by 1 or 2 ULP. This
    # unclean fix makes sure x is strictly within bounds.
    def eval(x):
        x = _check_clip_x(x, bounds)
        return func(x)

    return eval

def _check_clip_x(x, bounds):
    if (x < bounds[0]).any() or (x > bounds[1]).any():
        warnings.warn("Values in x were outside bounds during a "
                      "minimize step, clipping to bounds",
                      RuntimeWarning, stacklevel=3)
        x = np.clip(x, bounds[0], bounds[1])
        return x

    return x

def sequential_least_squares_programming_optimization(func, vars, x0, eqcons=(), ieqcons=(),
               bounds=(), fprime=None, args=(), iter=100, acc=1.0E-6,
               epsilon=_epsilon):

    eqcons_funcs = [sp.lambdify(vars, con, 'numpy') for con in eqcons]
    ieqcons_funcs = [sp.lambdify(vars, -con, 'numpy') for con in ieqcons]


    opts = {'maxiter': iter,
            'ftol': acc,
            'eps': epsilon,}

    # Build the constraints as a tuple of dictionaries
    cons = ()

    cons += tuple({'type': 'eq', 'fun': c, 'args': args} for c in eqcons_funcs)
    cons += tuple({'type': 'ineq', 'fun': c, 'args': args} for c in ieqcons_funcs)


    res, fun= _minimize_slsqp(func, vars, x0, jac=fprime, bounds=bounds,
                          constraints=cons, **opts)
    return res, fun


def _minimize_slsqp(func, vars, x0, jac=None, bounds=None,
                    constraints=(),
                    maxiter=100, ftol=1.0E-6,
                    eps=_epsilon, finite_diff_rel_step=None,
                    **unknown_options):
    iter = maxiter - 1
    acc = ftol

    # Transform x0 into an array.
    x = np.array(x0)

    # SLSQP is sent 'old-style' bounds, 'new-style' bounds are required by
    # ScalarFunction
    if bounds is None or len(bounds) == 0:
        new_bounds = (-np.inf, np.inf)
    else:
        new_bounds = old_bound_to_new(bounds)

    # clip the initial guess to bounds, otherwise ScalarFunction doesn't work
    x = np.clip(x, new_bounds[0], new_bounds[1])

    # Constraints are triaged per type into a dictionary of tuples
    if isinstance(constraints, dict):
        constraints = (constraints, )

    cons = {'eq': (), 'ineq': ()}
    for ic, con in enumerate(constraints):
        # check type
        try:
            ctype = con['type'].lower()
        except KeyError as e:
            raise KeyError('Constraint %d has no type defined.' % ic) from e
        except TypeError as e:
            raise TypeError('Constraints must be defined using a '
                            'dictionary.') from e
        except AttributeError as e:
            raise TypeError("Constraint's type must be a string.") from e
        else:
            if ctype not in ['eq', 'ineq']:
                raise ValueError("Unknown constraint type '%s'." % con['type'])

        # check function
        if 'fun' not in con:
            raise ValueError('Constraint %d has no function defined.' % ic)

        # check Jacobian
        cjac = con.get('jac')
        if cjac is None:
            # approximate Jacobian function. The factory function is needed
            # to keep a reference to `fun`, see gh-4240.
            def cjac_factory(fun, vars):
                def cjac(x, *args):
                    x = _check_clip_x(x, new_bounds)

                    return difftools.jacobian([fun], vars, x)

                return cjac
            cjac = cjac_factory(con['fun'], vars)

        # update constraints' dictionary
        cons[ctype] += ({'fun': con['fun'],
                         'jac': cjac,
                         'args': con.get('args', ())}, )

    exit_modes = {-1: "Gradient evaluation required (g & a)",
                   0: "Optimization terminated successfully",
                   1: "Function evaluation required (f & c)",
                   2: "More equality constraints than independent variables",
                   3: "More than 3*n iterations in LSQ subproblem",
                   4: "Inequality constraints incompatible",
                   5: "Singular matrix E in LSQ subproblem",
                   6: "Singular matrix C in LSQ subproblem",
                   7: "Rank-deficient equality constraint subproblem HFTI",
                   8: "Positive directional derivative for linesearch",
                   9: "Iteration limit reached"}

    # Set the parameters that SLSQP will need
    # meq, mieq: number of equality and inequality constraints

    meq = len(cons['eq'])
    mieq = len(cons['ineq'])
    # m = The total number of constraints
    m = meq + mieq
    # la = The number of constraints, or 1 if there are no constraints
    la = array([1, m]).max()
    # n = The number of independent variables
    n = len(x)

    # Define the workspaces for SLSQP
    n1 = n + 1
    mineq = m - meq + n1 + n1
    len_w = (3*n1+m)*(n1+1)+(n1-meq+1)*(mineq+2) + 2*mineq+(n1+mineq)*(n1-meq) \
            + 2*meq + n1 + ((n+1)*n)//2 + 2*m + 3*n + 3*n1 + 1
    len_jw = mineq
    w = zeros(len_w)
    jw = zeros(len_jw)

    # Decompose bounds into xl and xu
    if bounds is None or len(bounds) == 0:
        xl = np.empty(n, dtype=float)
        xu = np.empty(n, dtype=float)
        xl.fill(np.nan)
        xu.fill(np.nan)
    else:
        bnds = array([(_arr_to_scalar(l), _arr_to_scalar(u))
                      for (l, u) in bounds], float)
        if bnds.shape[0] != n:
            raise IndexError('SLSQP Error: the length of bounds is not '
                             'compatible with that of x0.')

        with np.errstate(invalid='ignore'):
            bnderr = bnds[:, 0] > bnds[:, 1]

        if bnderr.any():
            raise ValueError('SLSQP Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))
        xl, xu = bnds[:, 0], bnds[:, 1]

        # Mark infinite bounds with nans; the Fortran code understands this
        infbnd = ~isfinite(bnds)
        xl[infbnd[:, 0]] = np.nan
        xu[infbnd[:, 1]] = np.nan

    f_lamda = sp.lambdify(vars, func, 'numpy')

    def f_wrapper(x):
        return f_lamda(*x)

    def df_wrapper(x):
        return np.array(difftools.gradient(func, vars, x))

    # gh11403 SLSQP sometimes exceeds bounds by 1 or 2 ULP, make sure this
    # doesn't get sent to the func/grad evaluator.
    wrapped_fun = _clip_x_for_func(f_wrapper, new_bounds)
    wrapped_grad = _clip_x_for_func(df_wrapper, new_bounds)

    # Initialize the iteration counter and the mode value
    mode = array(0, int)
    acc = array(acc, float)
    majiter = array(iter, int)
    majiter_prev = 0

    # Initialize internal SLSQP state variables
    alpha = array(0, float)
    f0 = array(0, float)
    gs = array(0, float)
    h1 = array(0, float)
    h2 = array(0, float)
    h3 = array(0, float)
    h4 = array(0, float)
    t = array(0, float)
    t0 = array(0, float)
    tol = array(0, float)
    iexact = array(0, int)
    incons = array(0, int)
    ireset = array(0, int)
    itermx = array(0, int)
    line = array(0, int)
    n1 = array(0, int)
    n2 = array(0, int)
    n3 = array(0, int)

    # mode is zero on entry, so call objective, constraints and gradients
    # there should be no func evaluations here because it's cached from
    # ScalarFunction
    fx = wrapped_fun(x)
    g = append(wrapped_grad(x), 0.0)
    c = _eval_constraint(x, cons)
    a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

    while 1:
        # Call SLSQP
        slsqp.slsqp(m, meq, x, xl, xu, fx, c, g, a, acc, majiter, mode, w, jw,
              alpha, f0, gs, h1, h2, h3, h4, t, t0, tol,
              iexact, incons, ireset, itermx, line,
              n1, n2, n3)

        if mode == 1:  # objective and constraint evaluation required
            fx = wrapped_fun(x)
            c = _eval_constraint(x, cons)

        if mode == -1:  # gradient evaluation required
            g = append(wrapped_grad(x), 0.0)
            a = _eval_con_normals(x, cons, la, n, m, meq, mieq)

        # If exit mode is not -1 or 1, slsqp has completed
        if abs(mode) != 1:
            break

        majiter_prev = int(majiter)


    if mode == 0:
        print(f"SLSQP optimization convergence achieved after {majiter_prev} iterations")
    # elif mode == 8:
    #     print(f"SLSQP optimization convergence at bound achieved after {majiter_prev} iterations")
    else:
        warnings.warn("Problem solving may not have been successful, check the model.",
                      RuntimeWarning, stacklevel=3)
        warnings.warn(exit_modes[int(mode)] + "    (Exit mode " + str(mode) + ')',
                      RuntimeWarning, stacklevel=3)
    return x, fx


def _eval_constraint(x, cons):
    # Compute constraints
    if cons['eq']:
        c_eq = concatenate([atleast_1d(con['fun'](*x))
                            for con in cons['eq']])
    else:
        c_eq = zeros(0)

    if cons['ineq']:
        c_ieq = concatenate([atleast_1d(con['fun'](*x))
                             for con in cons['ineq']])
    else:
        c_ieq = zeros(0)

    # Now combine c_eq and c_ieq into a single matrix
    c = concatenate((c_eq, c_ieq))
    return c


def _eval_con_normals(x, cons, la, n, m, meq, mieq):
    # Compute the normals of the constraints
    if cons['eq']:
        a_eq = vstack([con['jac'](x, *con['args'])
                       for con in cons['eq']])
    else:  # no equality constraint
        a_eq = zeros((meq, n))

    if cons['ineq']:
        a_ieq = vstack([con['jac'](x, *con['args'])
                        for con in cons['ineq']])
    else:  # no inequality constraint
        a_ieq = zeros((mieq, n))

    # Now combine a_eq and a_ieq into a single a matrix
    if m == 0:  # no constraints
        a = zeros((la, n))
    else:
        a = vstack((a_eq, a_ieq))
    a = concatenate((a, zeros([la, 1])), 1)

    return a

if __name__ == '__main__':

    x1, x2 = sp.symbols('x1 x2')

    func = x1 ** 2 + x2 ** 2

    eqcons = [x1 + x2 - 2]
    ieqcons = [x1 - x2,x1+2]

    x0 = [0.5, 0.5]

    bounds = [(0, 1), (0, 1)]

    res = sequential_least_squares_programming_optimization(func, (x1, x2), x0, eqcons=eqcons, ieqcons=ieqcons, bounds=bounds)

    print(res)