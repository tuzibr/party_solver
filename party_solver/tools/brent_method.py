import numpy as np
import sympy as sp

def brent_method(func, lb=-500, ub=500, tol=1e-8, iter_num=100):
    a, c = lb, ub
    b = (a + c) / 2.0

    GOLD = (3 - np.sqrt(5)) / 2  # 黄金分割常数

    x = b
    w = b
    v = b

    fw = func(w)
    fx = fw
    fv = fw

    e = 0
    d = 0
    for i in range(iter_num):
        xm = (a + c) / 2.0
        tol1 = tol * abs(x) + np.finfo(float).eps
        tol2 = 2 * tol1

        if abs(x - xm) <= (tol2 - 0.5 * (c - a)):
            return x

        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            etemp = e
            e = d

            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (c - x):
                if x >= xm:
                    e = a - x
                else:
                    e = c - x
                d = GOLD * e
            else:
                d = p / q
                u = x + d
                if u - a < tol2 or c - u < tol2:
                    d = np.sign(tol1) * (xm - x)
        else:
            if x >= xm:
                e = a - x
            else:
                e = c - x
            d = GOLD * e

        if abs(d) >= tol1:
            u = x + d
        else:
            u = x + np.sign(tol1) * d

        fu = func(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                c = u
            if fu <= fw or w == x:
                v = w
                w = u
                fv = fw
                fw = fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

    return x


if __name__ == '__main__':

    alpha = sp.symbols('alpha')
    func = sp.lambdify(alpha, (alpha-2)**2)

    x_min = brent_method(func)
    print(f'min x = {x_min}，f(x) = {func(x_min)}')
