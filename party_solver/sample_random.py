# 导入random工具，用于生成随机数
from party_solver.tools import random
# 导入numpy库，用于计算均值和方差
import numpy as np

# 初始化随机数生成器
rng = random.Random()
# 设置随机数生成方法为线性同余生成器
rng.set_method('lcg')
# 设置种子
rng.set_seed(1234567)

# 生成1000个在-1到1之间的随机数
y = rng.generate(1000, lb=-1, ub=1)
# 输出生成的随机数数组
print('随机数列表',y)

# 计算随机数数组y的均值
mean_y = np.mean(y)
# 计算随机数数组y的方差
variance_y = np.var(y)
# 输出均值和方差
print(f'均值: {mean_y}')
print(f'方差: {variance_y}')

