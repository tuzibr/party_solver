import numpy as np
from sympy import symbols, Ge, Le, Gt, Lt, Eq, Add, collect, S, Max, Mul,Piecewise,floor,ceiling
import sympy as sp
import heapq
import matplotlib.pyplot as plt
from party_solver.solver import solver_lp
from party_solver.solver import solver_bc
from party_solver.solver import solver_ie
from party_solver.solver import solver_p
from party_solver.solver import solver_alm
from party_solver.solver import solver_sqp
from party_solver.solver import solver_ip
from party_solver.tools import gold_method
from party_solver.solver import solver_lbfgsb
import warnings
import time

class OptimizeWarning(UserWarning):
    pass

class Model:
    def __init__(self, model_name=None):
        # 初始化模型，存储变量字典和维护变量顺序列表
        self._model_name = model_name
        self._variables = {}  # 储存计算变量变量字典
        self._variable_order = []  # 维护计算变量顺序
        self._variables_int = []  # 储存整数列表
        self._variables_binary = []  # 储存二元变量列表
        self._set_variables = {}  # 原始变量字典
        self._decomposed_variables = {}  # 储存分解变量字典
        self._set_variable_order = []  # 维护原始变量顺序
        self._symbolic_variables = {}  # 存储符号变量
        self._constraints = []
        self._objective = None
        self._objective_sense = 'min'
        self._solution_var = None
        self._objective_var = None
        self._constant_term = []
        self._is_linear = False
        self._non_linear_objective = 0
        self._constraints_expression = []
        self._penalty_function = 0
        self._variable_tuple = ()
        self._la_objective = 0
        self._la_constraints = 0
        self._phr_objective = 0
        self._equality_constraints = []
        self._inequality_constraints = []
        self._lamda = []

        # 初始化堆栈
        self._solution_stack = []
        self._unique_solutions = []

        # Initialize c, A, b
        self._c = np.array([])  # 目标函数
        self._A = np.zeros((0, 0))  # 约束矩阵
        self._b = np.array([])  # RHS矩阵
        self._S = np.zeros((0, 0))  # 松弛变量矩阵
        self._B = np.zeros((0, 0))  # 人工变量矩阵
        self._E = np.array([])  # 符号矩阵
        self._lower_bounds = []  # 存储变量的下界
        self._upper_bounds = []  # 存储变量的上界
        self._initial_values = []  # 存储变量的初始值

        # 参数
        self.method = None
        self.sub_method = None
        self.sigma = 10
        self.max_iter = 100
        self.tol = 1e-6
        self.num_particles = 50
        self.max_iter_pso = 200
        self.w_max = 0.9
        self.w_min = 0.4
        self.cognitive_start = 2.5
        self.cognitive_end = 0.5
        self.social_start = 0.5
        self.social_end = 2.5

    def add_variable(self, lb=0, ub=float('inf'), vtype='continuous', name=None, flag=True, initial_value=None):
        if ub<=lb:
            raise ValueError(
                'The upper bound must be greater than the lower bound.')
        random_method = (
                    self.sub_method == 'pso' or self.sub_method == 'dbo')
        bound_method = (
                    self.sub_method == 'pso' or self.sub_method == 'dbo' or self.sub_method == 'lbfgsb' or self.method == 'slsqp' or self.sub_method == None)
        if self.method == 'ip':
            bound_method = False

        if self._is_linear is False and ub == float('inf') and initial_value is None:
            raise ValueError(
                'The upper bound of the variable must be finite or set initial value when the model is non-linear.')

        # 自动命名
        if name is None:
            name = f'var_{len(self._variables)}'
            warnings.warn(f"Variable was not named. Auto-generated name is '{name}'.", RuntimeWarning, stacklevel=3)

        if name in self._variables:
            raise ValueError(f"Variable name '{name}' is already used. Variable addition failed.")

        if flag is True:
            # 添加变量进原始变量计算字典
            self._set_variables[name] = {'lb': lb, 'ub': ub, 'type': vtype}
            self._set_variable_order.append(name)

        # 为变量创建符号
        var_symbol = symbols(name)
        self._symbolic_variables[name] = var_symbol
        # print(self._symbolic_variables)

        if lb < 0:
            if self._is_linear is True:
                # 需要分解变量
                positive_var_name = f"{name}_plus"
                negative_var_name = f"{name}_minus"

                # 添加两个非负变量
                self._add_positive_negative_variables(positive_var_name, negative_var_name, ub, -lb, vtype)

                # 存储分解变量与原变量的映射关系
                self._decomposed_variables[name] = (positive_var_name, negative_var_name)
                return var_symbol
            else:
                # 添加变量进参与计算字典
                self._variables[name] = {'lb': lb, 'ub': ub, 'type': vtype}

                # if vtype == 'integer':
                #     self._variables_int.append(len(self._variable_order))
                # elif vtype == 'binary':
                #     self._variables_binary.append(len(self._variable_order))
                # elif vtype == 'continuous':
                #     pass
                # else:
                #     raise ValueError(f"Invalid variable type '{vtype}'.")

                self._variable_order.append(name)

                # # 更新c向量以容纳新变量
                # self._c = np.append(self._c, 0)  # 初始假设系数为0
                #
                # # 更新矩阵A的维度以容纳新变量
                # num_constraints = self._A.shape[0]
                # new_column = np.zeros((num_constraints, 1))  # 为A创建新列
                # self._A = np.hstack((self._A, new_column))  # 将新列添加到A

                if self._is_linear is False:
                    if random_method is True:
                        if ub == float('inf'):
                            raise ValueError(f'Set bounds when use {self.sub_method}.')
                    else:
                        if initial_value is None:
                            if ub == float('inf') or lb == -float('inf'):
                                raise ValueError('Set initial value.')
                            else:
                                pass

                    if initial_value is not None:
                        self._initial_values.append(initial_value)

                # 设置变量的下界和上界
                self._lower_bounds.append(lb)
                self._upper_bounds.append(ub)

                add_flag = (self._is_linear or bound_method)

                if flag is False:
                    # 添加变量的上界约束，如果上界不是无限大
                    if ub != float('inf'):
                        self.add_constraint(Le(var_symbol, ub))
                    if lb != 0:
                        self.add_constraint(Ge(var_symbol, lb))

                if add_flag is False:
                    if lb != -float('inf'):
                        self.add_constraint(Ge(var_symbol, lb))
                    if ub != float('inf'):
                        self.add_constraint(Le(var_symbol, ub))

                return var_symbol
        else:
            # 不需要分解，直接添加

            # 添加变量进参与计算字典
            self._variables[name] = {'lb': lb, 'ub': ub, 'type': vtype}
            if vtype == 'integer':
                self._variables_int.append(len(self._variable_order))
            elif vtype == 'binary':
                self._variables_binary.append(len(self._variable_order))
            elif vtype == 'continuous':
                pass
            else:
                raise ValueError(f"Invalid variable type '{vtype}'.")

            self._variable_order.append(name)

            # 更新c向量以容纳新变量
            self._c = np.append(self._c, 0)  # 初始假设系数为0

            # 更新矩阵A的维度以容纳新变量
            num_constraints = self._A.shape[0]
            new_column = np.zeros((num_constraints, 1))  # 为A创建新列
            self._A = np.hstack((self._A, new_column))  # 将新列添加到A

            if self._is_linear is False:

                if random_method is True:
                    if ub == float('inf'):
                        raise ValueError(f'Set bounds when use {self.sub_method}.')
                else:
                    if initial_value is None:
                        if ub == float('inf') or lb == -float('inf'):
                            raise ValueError('Set initial value.')
                        else:
                            pass

                if initial_value is not None:
                    self._initial_values.append(initial_value)

            # 设置变量的下界和上界
            self._lower_bounds.append(lb)
            self._upper_bounds.append(ub)

            add_flag = (self._is_linear or bound_method)

            if flag is False:
                # 添加变量的上界约束，如果上界不是无限大
                if ub != float('inf'):
                    self.add_constraint(Le(var_symbol, ub))
                if lb != 0:
                    self.add_constraint(Ge(var_symbol, lb))

            if add_flag is False:
                if lb != -float('inf'):
                    self.add_constraint(Ge(var_symbol, lb))
                if ub != float('inf'):
                    self.add_constraint(Le(var_symbol, ub))

            return var_symbol

    def _add_positive_negative_variables(self, positive_var_name, negative_var_name, positive_var_ub=float('inf'),
                                         negative_var_ub=float('inf'), atype='continuous'):
        self.add_variable(name=positive_var_name, lb=0, ub=positive_var_ub, vtype=atype, flag=False)
        self.add_variable(name=negative_var_name, lb=0, ub=negative_var_ub, vtype=atype, flag=False)

    def add_constraint(self, expr, name=None):
        # # 遍历所有需要替换的变量的映射，并应用替换
        # for original_var, (pos_var, neg_var) in self._decomposed_variables.items():
        #     replacement_expr = {symbols(original_var): symbols(pos_var) - symbols(neg_var)}
        #     expr = expr.subs(replacement_expr)
        if name is not None:
            pass
        if self._is_linear is True:
            # 获取表达式中的所有变量
            expr_variables = expr.free_symbols

            # 遍历所有需要替换的变量的映射
            for original_var, (pos_var, neg_var) in self._decomposed_variables.items():
                original_sym = symbols(original_var)  # 将原始变量名转换为符号
                if original_sym in expr_variables:  # 只处理出现在表达式中的变量
                    # 创建替换表达式
                    pos_sym = symbols(pos_var)
                    neg_sym = symbols(neg_var)
                    replacement_expr = {original_sym: pos_sym - neg_sym}
                    # 应用替换
                    expr = expr.subs(replacement_expr)

            # 检查约束类型（不等式还是等式）
            if isinstance(expr, (Ge, Gt)):  # 大于等于或大于
                normalized_expr = expr.rhs - expr.lhs
                constraint_type = 'Ge'
            elif isinstance(expr, (Le, Lt)):  # 小于等于或小于
                normalized_expr = expr.lhs - expr.rhs
                constraint_type = 'Le'
            elif isinstance(expr, Eq):  # 等式
                normalized_expr = expr.lhs - expr.rhs
                constraint_type = 'equality'
            else:
                raise ValueError("不支持的约束表达式类型。")

            # 获取表达式中的所有变量
            variables = list(normalized_expr.free_symbols)
            # 获取常数项
            # 使用collect来提取每个变量的系数并保留常数项
            collected_expr = collect(normalized_expr, variables, evaluate=False)
            coefficients_dict = {v: collected_expr.get(v, S.Zero) for v in variables}
            constant_term = collected_expr.get(S.One, S.Zero)

            # 创建新的约束行
            new_row = np.zeros(len(self._variable_order))
            for variable, coeff in coefficients_dict.items():
                if variable.name in self._variable_order:
                    index = self._variable_order.index(variable.name)
                    new_row[index] = coeff

            # 更新矩阵
            if constant_term <= 0:
                self._A = np.vstack([self._A, new_row])
                self._b = np.append(self._b, -constant_term)
                self._E = np.append(self._E, 1)
            else:
                self._A = np.vstack([self._A, -new_row])
                self._b = np.append(self._b, constant_term)
                self._E = np.append(self._E, -1)

            self._B = np.vstack([self._B, np.zeros((1, self._B.shape[1]))])
            self._S = np.vstack([self._S, np.zeros((1, self._S.shape[1]))])

            # 管理人工变量矩阵B
            if constraint_type == 'equality':
                self._B = np.hstack([self._B, np.zeros((self._B.shape[0], 1))])  # 增加一列
                self._B[-1, -1] = 1  # 在新增列的最后一行设置为1
                self._E[-1] = 0

            if constant_term > 0:
                if constraint_type != 'equality':
                    self._B = np.hstack([self._B, np.zeros((self._B.shape[0], 1))])  # 增加一列
                self._B[-1, -1] = 1  # 在新增列的最后一行设置为1

            # 管理松弛变量矩阵S
            if constraint_type != 'equality':
                self._S = np.hstack([self._S, np.zeros((self._S.shape[0], 1))])  # 增加一列
                if constant_term <= 0:
                    self._S[-1, -1] = 1  # 在新增列的最后一行设置为1
                else:
                    self._S[-1, -1] = -1
            else:
                self._S = np.hstack([self._S, np.zeros((self._S.shape[0], 1))])  # 增加一列

        else:
            self._constraints_expression.append(expr)
            # 检查约束类型（不等式还是等式）
            if isinstance(expr, (Ge, Gt)):  # 大于等于或大于
                normalized_expr = Max(0, expr.rhs - expr.lhs) ** 2
                self._inequality_constraints.append(expr.rhs - expr.lhs)
            elif isinstance(expr, (Le, Lt)):  # 小于等于或小于
                normalized_expr = Max(0, expr.lhs - expr.rhs) ** 2
                self._inequality_constraints.append(expr.lhs - expr.rhs)
            elif isinstance(expr, Eq):  # 等式
                normalized_expr = (expr.lhs - expr.rhs) ** 2
                self._equality_constraints.append(expr.lhs - expr.rhs)
            else:
                raise ValueError("不支持的约束表达式类型。")
            # print(normalized_expr)
            self._penalty_function += normalized_expr

    def set_objective(self, expr, sense='min'):

        # p = sp.Poly(expr)  # 显式地将表达式转换为多项式
        # degree = p.total_degree()
        # # 检查表达式是否为线性
        # self._is_linear = expr.is_polynomial() and degree == 1

        # 处理线性目标函数
        if self._is_linear:
            # 获取表达式中的所有变量
            expr_variables = expr.free_symbols
            # 遍历所有需要替换的变量的映射
            for original_var, (pos_var, neg_var) in self._decomposed_variables.items():
                original_sym = symbols(original_var)  # 将原始变量名转换为符号
                if original_sym in expr_variables:  # 只处理出现在表达式中的变量
                    # 创建替换表达式
                    pos_sym = symbols(pos_var)
                    neg_sym = symbols(neg_var)
                    replacement_expr = {original_sym: pos_sym - neg_sym}
                    # 应用替换
                    expr = expr.subs(replacement_expr)

            # 初始化目标函数系数向量，确保长度与变量顺序一致
            self._c = np.zeros(len(self._variable_order))

            # 获取表达式中的所有变量
            variables = list(expr.free_symbols)
            # 提取每个变量的系数
            coefficients_dict = {v: expr.coeff(v) for v in variables}
            # 获取常数项
            self._constant_term = expr.as_coeff_add()[0]
            # 遍历变量顺序列表，更新目标系数向量c
            for variable, coeff in coefficients_dict.items():
                if variable.name in self._variable_order:
                    index = self._variable_order.index(variable.name)
                    self._c[index] = coeff

            # 设置求解目标是最小化还是最大化
            self._objective_sense = sense
            # 如果是最大化问题，将系数向量取负
            if self._objective_sense != 'min':
                self._c = -self._c

        else:
            # 设置求解目标是最小化还是最大化
            self._objective_sense = sense

            self._constant_term = 0
            # 处理非线性目标函数
            self._c = None  # 非线性情况下不适用线性系数向量
            if self._objective_sense == 'min':
                self._non_linear_objective = expr
            else:
                self._non_linear_objective = -expr

        # # 确保 A 矩阵的列数与c向量长度一致
        # if self._A.shape[1] != len(self._c):
        #     # 扩展 A 矩阵的列数
        #     new_columns = np.zeros((self._A.shape[0], len(self._c) - self._A.shape[1]))
        #     self._A = np.hstack((self._A, new_columns))

        # 打印或调试信息
        # print("Updated objective coefficients (c vector):", self._c)
        # print("Updated A matrix size:", self._A.shape)

    def optimize(self, num_runs=1,time_flag=True):
        if self.method == None:
            if len(self._set_variable_order) >= 5000:
                if self.sub_method == None:
                    self.method = 'alm'
                    self.sub_method = 'lbfgsb'
                else:
                    self.method = 'alm'

                print("The number of variables is large, use default alm.")

        start_time = time.time()

        # 线性规划
        if np.any(self._c) != 0 and self._is_linear is True:
            # 检查所有变量的类型
            if any(var['type'] == 'binary' for var in self._variables.values()):
                self._solution_var = solver_ie.implicit_enumeration(self._c, self._A, self._b, self._S, self._B,
                                                                    self._E, self._lower_bounds, self._upper_bounds,
                                                                    self._variables_int, self._variables_binary)

            elif any(var['type'] == 'integer' for var in self._variables.values()):
                self._solution_var = \
                    solver_bc.branch_and_bound(self._c, self._A, self._b, self._S, self._B, self._E, self._lower_bounds,
                                               self._upper_bounds, self._variables_int)[0]

            else:
                self._solution_var = \
                    solver_lp.simplex_method(self._c, self._A, self._b, self._S, self._B, self._E, self._lower_bounds,
                                             self._upper_bounds)[0]

        # 非线性规划
        elif self._is_linear is not True:

            if self.method == 'gold':
                # 获取表达式中的所有变量
                expr_variables = self._non_linear_objective.free_symbols

                objective_function = sp.lambdify(tuple(expr_variables), self._non_linear_objective, 'math')

                self._variable_tuple = tuple(expr_variables)
                # 使用黄金分割法寻找目标函数的最小值
                self._solution_var, intervals = gold_method.golden_section_search(objective_function,
                                                                                  self._lower_bounds[0],
                                                                                  self._upper_bounds[0], 1e-8)

                self._objective_var = objective_function(self._solution_var)

                # 绘制函数曲线
                x_values = [z * 0.01 for z in
                            range(int(self._lower_bounds[0] * 100), int(self._upper_bounds[0] * 100) + 1)]
                y_values = [objective_function(z) for z in x_values]

                plt.plot(x_values, y_values, label='function')
                plt.axhline(0, color='black', linewidth=0.5)
                plt.axvline(0, color='black', linewidth=0.5)
                plt.grid(color='gray', linestyle='--', linewidth=0.5)

                # 绘制搜索区间的变化
                for (a, b) in intervals:
                    plt.plot([a, b], [objective_function(a), objective_function(b)], 'ro-')

                plt.title('Golden Section Search')
                plt.xlabel(tuple(expr_variables))
                plt.ylabel('function value')
                plt.legend()
                plt.show()

                self._solution_var = [self._solution_var]
                return None
            else:
                # print(self._non_linear_objective)
                # print(self._penalty_function)
                # 提取字典中的值并转换为元组
                self._variable_tuple = tuple(self._symbolic_variables.values())
                # if self.method == 'p':
                #     for i in range(num_runs):
                #         try:
                #             solution_var, objective_var, penalty_value, _, _ = solver_p.calmin(
                #                 self._non_linear_objective,
                #                 self._penalty_function,
                #                 list(zip(self._lower_bounds, self._upper_bounds)),
                #                 self._variable_tuple,
                #                 sigma=self.sigma,
                #                 max_iter=self.max_iter,
                #                 tol=self.tol,
                #                 num_particles=self.num_particles,
                #                 max_iter_pso=self.max_iter_pso,
                #                 w_max=self.w_max,
                #                 w_min=self.w_min,
                #                 cognitive_start=self.cognitive_start,
                #                 cognitive_end=self.cognitive_end,
                #                 social_start=self.social_start,
                #                 social_end=self.social_end
                #             )
                #
                #         except Warning:
                #             solution_var, objective_var, penalty_value, _, _ = solver_p.calmin(
                #                 self._non_linear_objective,
                #                 self._penalty_function,
                #                 list(zip(self._lower_bounds, self._upper_bounds)),
                #                 self._variable_tuple,
                #                 sigma=self.sigma,
                #                 max_iter=self.max_iter,
                #                 tol=self.tol,
                #                 num_particles=self.num_particles,
                #                 max_iter_pso=self.max_iter_pso,
                #                 w_max=self.w_max,
                #                 w_min=self.w_min,
                #                 cognitive_start=self.cognitive_start,
                #                 cognitive_end=self.cognitive_end,
                #                 social_start=self.social_start,
                #                 social_end=self.social_end
                #             )
                #             pass
                #
                #         solution_dict = dict(zip(self._variable_tuple, solution_var))
                #         self._la_objective = self._non_linear_objective
                #         self._la_constraints = 0
                #
                #         m = 1e8
                #         for expr in self._constraints_expression:
                #             if isinstance(expr, (Ge, Gt)):  # 大于等于或大于
                #                 normalized_expr = expr.rhs - expr.lhs
                #                 if -1e-3 <= normalized_expr.subs(solution_dict) <= 1e-3:
                #                     self._la_constraints += m * normalized_expr ** 2
                #             elif isinstance(expr, (Le, Lt)):  # 小于等于或小于
                #                 normalized_expr = expr.lhs - expr.rhs
                #                 if -1e-3 <= normalized_expr.subs(solution_dict) <= 1e-3:
                #                     self._la_constraints += m * normalized_expr ** 2
                #             else:
                #                 normalized_expr = expr.lhs - expr.rhs
                #                 if -1e-3 <= normalized_expr.subs(solution_dict) <= 1e-3:
                #                     self._la_constraints += m * normalized_expr ** 2
                #         #
                #         self._la_objective += self._la_constraints
                #
                #         try:
                #             solution_var1, objective_var1, penalty_value1, _, _ = solver_p.calmin(
                #                 self._non_linear_objective,
                #                 self._penalty_function,
                #                 [(value - 0.1, value + 0.1) for value in solution_var],
                #                 self._variable_tuple,
                #                 sigma=self.sigma,
                #                 max_iter=self.max_iter,
                #                 tol=self.tol,
                #                 num_particles=self.num_particles,
                #                 max_iter_pso=self.max_iter_pso,
                #                 w_max=0.4,
                #                 w_min=0.1,
                #                 cognitive_start=self.cognitive_end,
                #                 cognitive_end=self.cognitive_end,
                #                 social_start=self.social_end,
                #                 social_end=self.social_end
                #             )
                #             if objective_var1 < objective_var:
                #                 solution_var = solution_var1
                #                 objective_var = objective_var1
                #                 penalty_value = penalty_value1
                #         except Warning:
                #             pass
                #
                #         if num_runs != 1:
                #             print(f'[Solution {i + 1}]')
                #             print('[Current Value]', objective_var)
                #             print('[Current Solution]', solution_var)
                #             print('-'*1000)
                #             print('')
                #         if penalty_value <= 1e-8 or penalty_value == np.inf:
                #             # 将解推入堆栈
                #             solution_tuple = tuple(solution_var)  # 将数组转换为元组
                #             heapq.heappush(self._solution_stack, (objective_var, solution_tuple))

                if self.method == 'alm' or (self.sub_method != None and self.method == None):
                    self.method = 'alm'
                    for i in range(num_runs):
                        self._phr_objective = self._non_linear_objective
                        solution_var, objective_var = solver_alm.calmin(self._phr_objective,
                                                                        self._equality_constraints,
                                                                        self._inequality_constraints,
                                                                        list(zip(self._lower_bounds,
                                                                                 self._upper_bounds)),
                                                                        self._variable_tuple,
                                                                        x0=self._initial_values,
                                                                        tol=self.tol,
                                                                        method=self.sub_method,
                                                                        num_particles=self.num_particles,
                                                                        max_iter_pso=self.max_iter_pso,
                                                                        w_max=0.4,
                                                                        w_min=0.1,
                                                                        cognitive_start=self.cognitive_end,
                                                                        cognitive_end=self.cognitive_end,
                                                                        social_start=self.social_end,
                                                                        social_end=self.social_end
                                                                        )

                        if num_runs != 1:
                            print(f'[Solution {i + 1}]')
                            print('[Current Value]', objective_var)
                            print('[Current Solution]', solution_var)
                            print('-'*1000)
                        # 将解推入堆栈
                        solution_tuple = tuple(solution_var)  # 将数组转换为元组
                        heapq.heappush(self._solution_stack, (objective_var, solution_tuple))
                elif self.method == 'p' :
                    self.method = 'p'
                    for i in range(num_runs):
                        self._phr_objective = self._non_linear_objective
                        solution_var, objective_var = solver_p.calmin(self._phr_objective,
                                                                        self._equality_constraints,
                                                                        self._inequality_constraints,
                                                                        list(zip(self._lower_bounds,
                                                                                 self._upper_bounds)),
                                                                        self._variable_tuple,
                                                                        sigma=self.sigma,
                                                                        x0=self._initial_values,
                                                                        tol=self.tol,
                                                                        method=self.sub_method,
                                                                        num_particles=self.num_particles,
                                                                        max_iter_pso=self.max_iter_pso,
                                                                        w_max=0.4,
                                                                        w_min=0.1,
                                                                        cognitive_start=self.cognitive_end,
                                                                        cognitive_end=self.cognitive_end,
                                                                        social_start=self.social_end,
                                                                        social_end=self.social_end
                                                                        )

                        if num_runs != 1:
                            print(f'[Solution {i + 1}]')
                            print('[Current Value]', objective_var)
                            print('[Current Solution]', solution_var)
                            print('-'*1000)
                        # 将解推入堆栈
                        solution_tuple = tuple(solution_var)  # 将数组转换为元组
                        heapq.heappush(self._solution_stack, (objective_var, solution_tuple))
                elif self.method == 'ip':
                    full = len(self._inequality_constraints)>0
                    if len(self._initial_values) ==0:
                        warnings.warn('Set initial value as middle of bounds.', RuntimeWarning, stacklevel=3)
                        self._initial_values = np.array([(x + y) / 2 for x, y in list(zip(self._lower_bounds, self._upper_bounds))])
                    if full:
                        solution_var, objective_var = solver_ip.interior_point_method(self._non_linear_objective,
                                                                            self._variable_tuple,
                                                                            self._initial_values,
                                                                            self._equality_constraints,
                                                                            self._inequality_constraints,
                                                                            tol=self.tol,
                                                                            max_iter=self.max_iter,
                                                                            )
                    else:
                        raise ValueError("Use Interior Point only in the presence of inequality constraints.")
                    solution_tuple = tuple(solution_var)  # 将数组转换为元组
                    heapq.heappush(self._solution_stack, (objective_var, solution_tuple))
                elif self.method == 'slsqp':
                    full = len(self._equality_constraints)>0 or len(self._inequality_constraints)>0
                    if len(self._initial_values) ==0:
                        warnings.warn('Set initial value as middle of bounds.', RuntimeWarning, stacklevel=3)

                        self._initial_values = np.array([(x + y) / 2 for x, y in list(zip(self._lower_bounds, self._upper_bounds))])
                    if full:

                        solution_var, objective_var = solver_sqp.sequential_least_squares_programming_optimization(
                            self._non_linear_objective,
                            self._variable_tuple,
                            self._initial_values,
                            self._equality_constraints,
                            self._inequality_constraints,
                            list(zip(self._lower_bounds, self._upper_bounds)),
                            acc=self.tol
                        )
                    else:
                        raise ValueError("Use SLSQP only in the presence of constraints.")
                    solution_tuple = tuple(solution_var)  # 将数组转换为元组
                    heapq.heappush(self._solution_stack, (objective_var, solution_tuple))
                elif self.method == None:
                    full = len(self._equality_constraints)>0 or len(self._inequality_constraints)>0
                    if len(self._initial_values) ==0:
                        warnings.warn('Set initial value as middle of bounds.', RuntimeWarning, stacklevel=3)

                        self._initial_values = np.array([(x + y) / 2 for x, y in list(zip(self._lower_bounds, self._upper_bounds))])
                    else:
                        self._initial_values = np.array(self._initial_values)

                    if full:

                        self.method = 'slsqp'
                        solution_var, objective_var = solver_sqp.sequential_least_squares_programming_optimization(
                            self._non_linear_objective,
                            self._variable_tuple,
                            self._initial_values,
                            self._equality_constraints,
                            self._inequality_constraints,
                            list(zip(self._lower_bounds, self._upper_bounds)),
                            acc=self.tol
                        )
                    else:
                        self.sub_method = 'lbfgsb'
                        solution_var = solver_lbfgsb.l_bfgs_b_optimization(self._non_linear_objective, self._variable_tuple, self._initial_values, epsilon=self.tol,
                                                                          bounds=list(zip(self._lower_bounds, self._upper_bounds)))
                        function_callable = sp.lambdify(self._variable_tuple, self._non_linear_objective)
                        objective_var = function_callable(*solution_var)


                    solution_tuple = tuple(solution_var)  # 将数组转换为元组
                    heapq.heappush(self._solution_stack, (objective_var, solution_tuple))

                else:
                    raise NotImplementedError("Method is not implemented")

            self._objective_var, self._solution_var = heapq.nsmallest(1, self._solution_stack)[0]
            end_time = time.time()
            execution_time = end_time - start_time

            def print_boxed_text(text):
                # 计算方框的宽度和高度
                box_width = len(text) + 4

                print('┌' + '─' * (box_width - 2) + '┐')

                print(f"│ {' ' * ((box_width - len(text) - 4) // 2)}{text}{' ' * ((box_width - len(text) - 4) // 2)} │")

                print('└' + '─' * (box_width - 2) + '┘')

            if time_flag:
                print_boxed_text(f"Execution time: {execution_time / num_runs} second")
        else:
            raise NotImplementedError("Model is not completed")

    def getvars(self):
        if self._is_linear is True:
            # 确保已经进行了优化，且存在解
            if self._solution_var is None:
                raise ValueError("No solution available. Please run optimize() first.")
            self._solution_var = [float(value) for value in self._solution_var]
            # 创建一个字典，包含变量名和对应的优化值
            var_values = {}
            for name in self._set_variable_order:
                if name in self._decomposed_variables:
                    # 如果变量被分解，获取对应的x+和x-变量名
                    pos_var, neg_var = self._decomposed_variables[name]
                    # 找出这些变量在variable_order中的索引
                    pos_index = self._variable_order.index(pos_var)
                    neg_index = self._variable_order.index(neg_var)
                    # 使用索引从解中获取值
                    pos_value = self._solution_var[pos_index]
                    neg_value = self._solution_var[neg_index]
                    # 计算原始变量的值
                    original_value = pos_value - neg_value
                    var_values[name] = original_value
                elif name not in sum(self._decomposed_variables.values(), ()):
                    # 如果变量未分解，并且不是分解变量的一部分，直接获取其索引并从解中提取其值
                    index = self._variable_order.index(name)
                    var_values[name] = self._solution_var[index]
        else:
            # 创建一个字典，包含变量名和对应的优化值
            var_values = {}
            for name, value in zip(self._set_variable_order, self._solution_var):
                var_values[name] = value

        return var_values

    def objval(self):
        if self._is_linear is True:
            if self._solution_var is not None:
                if self._objective_sense != 'min':
                    self._objective_var = np.dot(-self._c, self._solution_var) + self._constant_term
                    return self._objective_var
                else:
                    self._objective_var = np.dot(self._c, self._solution_var) + self._constant_term
                    return self._objective_var
            else:
                print('Optimal Failed')
                return
        else:
            if self._objective_sense != 'min':
                return -self._objective_var + self._constant_term
            else:
                return self._objective_var + self._constant_term

    def extract_local_optima_solutions(self):
        def is_different_solution(sol1, sol2):
            return any(abs(a - b) > 1e-3 for a, b in zip(sol1, sol2))

        for objective_var, solution_tuple in self._solution_stack:
            if not any(not is_different_solution(unique_solution, solution_tuple) for _, unique_solution in
                       self._unique_solutions):
                self._unique_solutions.append((objective_var, solution_tuple))

        self._unique_solutions.sort(key=lambda z: z[0])

        if len(self._unique_solutions) != 1 and len(self._unique_solutions) != 0:
            print("Local optimal solutions found:")
            for i, (obj, sol) in enumerate(self._unique_solutions, 1):
                print(f"Local optimal {i}:\n Objective: {obj}, Solution: {sol}")
        else:
            print("No local optimal solutions found.")

    def print_model(self):
        if self._is_linear is True:
            print("Model Variables:")
            for name, var in self._set_variables.items():
                if var['type'] == 'binary':
                    print(f"Variable {name}: Type={var['type']}, value=0, 1")
                else:
                    print(f"Variable {name}: Type={var['type']}, Bounds=({var['lb']}, {var['ub']})")

            # 格式化并打印约束
            print("\nConstraints:")
            variable_names = [name for name in self._variables]  # 使用已经定义的变量名
            for row, rhs, eq in zip(self._A, self._b, self._E):
                constraint_str = " + ".join(f"{coeff}*{name}" for coeff, name in zip(row, variable_names) if coeff != 0)
                if eq == 1:
                    print(f"{constraint_str} <= {rhs}")
                elif eq == 0:
                    print(f"{constraint_str} == {rhs}")
                elif eq == -1:
                    print(f"{constraint_str} >= {rhs}")

            # 格式化并打印目标函数
            print("\nObjective Function:")
            objective_str = " + ".join(f"{coeff}*{name}" for coeff, name in zip(self._c, variable_names) if coeff != 0)
            if self._objective_sense == 'min':
                print(f"Minimize: {objective_str}")
            else:
                print(f"Maximize: " + " + ".join(f"{-coeff}*{name}" for coeff, name in zip(self._c, variable_names) if
                                                 coeff != 0))  # 如果是最大化，则将系数取负

            print("")
        else:
            print("Model Variables:")
            for name, var in self._set_variables.items():
                print(f"Variable {name}: Type={var['type']}, Bounds=({var['lb']}, {var['ub']})")

            print("\nConstraints:")

            solution_dict = dict(zip(self._variable_tuple, self._solution_var))

            for expr in self._constraints_expression:
                if isinstance(expr, (Ge, Gt)):  # 大于等于或大于
                    normalized_expr = expr.rhs - expr.lhs
                    print(expr, 'Constraint value (should be negative):', float(normalized_expr.subs(solution_dict)))
                elif isinstance(expr, (Le, Lt)):  # 小于等于或小于
                    normalized_expr = expr.lhs - expr.rhs
                    print(expr, 'Constraint value (should be negative):', float(normalized_expr.subs(solution_dict)))
                else:
                    normalized_expr = expr.lhs - expr.rhs
                    expr = f'{expr.lhs} == {expr.rhs}'
                    print(expr, 'Constraint value (should be zero):', float(normalized_expr.subs(solution_dict)))

            print("\nObjective Function:")
            if self._objective_sense == 'min':
                print(f"Minimize: {self._non_linear_objective}")
            else:
                print(f"Maximize: {-self._non_linear_objective}")

            print("\nOptimization Methods:")
            constraint_optimization_method = self.method == 'slsqp' or self.method == 'ip'
            if not  constraint_optimization_method:
                if self.method != None:
                    print(f"Constraint Optimization Method: {self.method}")
                print(f"Unconstrained Optimization Method: {self.sub_method}")
            else:
                print(f"Constraint Optimization Method: {self.method}")


        print("")
        return None

    def solve_constrained_qp(self):
        pass

    def getmatrix(self):
        if self._is_linear is True:
            # 输出c, A, b, S, B作为列表
            print("c =np.array(", self._c.tolist(), ')')
            print("A =np.array(", self._A.tolist(), ')')
            print("b =np.array(", self._b.tolist(), ')')
            print("S =np.array(", self._S.tolist(), ')')
            print("B =np.array(", self._B.tolist(), ')')
            print("E =np.array(", self._E.tolist(), ')')
            print("lower_bounds =np.array(", self._lower_bounds, ')')
            print("upper_bounds =np.array(", self._upper_bounds, ')')
        else:
            print("Non-linear model")


    @staticmethod
    def quicksum(expressions):
        total = 0
        for expr in expressions:
            total = Add(total, expr)  # 使用 Add 来逐一添加表达式
        return total

    @staticmethod
    def quickmultiply(expressions):
        total = 1
        for expr in expressions:
            total = Mul(total, expr)  # 使用 Mul 来逐一相乘表达式
        return total

    @staticmethod
    def Eq(a, b):
        expr = Eq(a, b)
        return expr

    @staticmethod
    def round(number):
        print(number)
        if type(number)!=dict:
            return Piecewise(
                (floor(number), number - floor(number) < 0.5),
                (ceiling(number), True)
            )
        elif type(number)==dict:
            new_dict = {}
            for key, value in number.items():
                new_dict[key] = Piecewise(
                    (floor(value), value - floor(value) < 0.5),
                    (ceiling(value), True)
                )
            return new_dict
        else:
            raise TypeError("Invalid input type")


    def set_initial_values(self, initial_guess):
        self._initial_values = initial_guess

    def set_linear_params(self, linear=True):
        self._is_linear = linear

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} is not a valid parameter")

    def set_inspired_params(self, pop_size=30,epoch=100):
        self.num_particles = pop_size
        self.max_iter_pso = epoch

    def test_function(self, value):
        # 提取字典中的值并转换为元组
        variable_tuple = tuple(self._symbolic_variables.values())
        obj = sp.lambdify(variable_tuple, self._non_linear_objective)

        # 对等式约束和不等式约束分别进行lambdify并生成列表
        eq_constraints = [sp.lambdify(variable_tuple, c) for c in self._equality_constraints]
        ineq_constraints = [sp.lambdify(variable_tuple, ic) for ic in self._inequality_constraints]

        print('Test Objective Value:', obj(*value))

        # 计算等式约束值
        eq_values = [eq_constraints[i](*value) for i in range(len(self._equality_constraints))]
        print('Equality Constraint Values:', eq_values)

        # 计算不等式约束值
        ineq_values = [ineq_constraints[j](*value) for j in range(len(self._inequality_constraints))]
        print('Inequality Constraint Values:', ineq_values)


if __name__ == '__main__':
    pass