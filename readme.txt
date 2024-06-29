# party_solver

pip install sympy
pip install numpy
pip install matplotlib
=======================================================================================
优化模块（optimize）
=======================================================================================
base.py: 定义了优化算法的基础类和简洁的建模方式。
=======================================================================================
求解器模块（solver）
=======================================================================================
线性规划
solver_lp.py: 线性规划（Linear Programming）。
solver_bc.py: 分支切割法（Branch And Cut）。
solver_ie.py: 隐枚举法（Implicit Enumeration）。
=======================================================================================
无约束优化方法
solver_gd.py: 梯度下降法（Gradient Descent）。
solver_cg.py: 共轭梯度法（Conjugate Gradient Method）。
solver_dn.py: 阻尼牛顿法（Damped-Newton Method）。
solver_qn.py: 拟牛顿法（Quasi-Newton Method）。
solver_lbfgsb.py: 处理边界的低内存BFGS法（L-BFGS-B）。
solver_pm.py: 鲍威尔法（Powell Method）。
solver_tr.py: 使用Dog-leg求解子问题的信赖域法（Trust Region）。
solver_nm.py: Nelder-Mead单形替换法（Nelder-Mead Simplex Method）。
solver_pso.py: 粒子群优化（Particle Swarm Optimization）。
solver_dbo.py: 蜣螂优化（Dung Beetle Optimization）。
=======================================================================================
约束优化方法
solver_p.py: 惩罚函数法（Penalty Method）。
solver_alm.py: 增广拉格朗日乘子法（Augmented Lagrangian Method）。
solver_ip.py: 原始对偶内点法（Primal Dual Interior Point Method）
solver_sqp.py: 序列最小二乘二次规划（Sequential Least Squares Quadratic Programming）。
=======================================================================================
工具模块（tools）
brent_method.py: Brent法。
gold_method.py: 黄金分割法（Golden Section Method）。
wolfe.py: 非精确搜索的沃尔夫准则（Wolfe Conditions）。
difftools.py: 使用有限差分法生成梯度、海塞矩阵、雅各布矩阵工具。
random.py: 随机数生成工具。
