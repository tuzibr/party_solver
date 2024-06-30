import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

r = 1
l = 4.10011141
d = 2.35402956
a = 5


phi_0 = np.arccos(((r + l) ** 2 - d**2 + a**2) / (2 * a * (r + l)))
theta_crank = np.linspace(phi_0, phi_0 + np.pi / 2, 1000)


def calculate_theta_rocker(r, l, d, a, phi_i):
    rho_i = np.sqrt(a ** 2 + r ** 2 - 2 * r * a * np.cos(phi_i))
    alpha_i = np.arccos((rho_i ** 2 + d ** 2 - l ** 2) / (2 * rho_i * d))
    beta_i = np.arccos((rho_i ** 2 + a ** 2 - r ** 2) / (2*a * rho_i))

    theta_rocker = np.zeros_like(phi_i)
    for idx, i in enumerate(phi_i):
        if 0 <= i < np.pi:
            theta = np.pi - alpha_i[idx] - beta_i[idx]
        else:
            theta = np.pi - alpha_i[idx] + beta_i[idx]

        theta_rocker[idx] = theta

    return theta_rocker

theta_rocker = calculate_theta_rocker(r, l, d, a, theta_crank)

# Parameters for the psi_Ei curve
psi_0 = np.arccos(((r + l) ** 2 - d**2 - a**2) / (2 * a * d))  # Initial psi
psi_Ei = psi_0 + (2 / (3 * np.pi)) * (theta_crank - phi_0) ** 2

# 加载中文字符的字体
font_path = '仿宋_GB2312.TTF'
font_prop = FontProperties(fname=font_path, size=12)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta_crank), np.degrees(theta_rocker), label='实际曲线: $\\psi_{i}$')
plt.plot(np.degrees(theta_crank), np.degrees(psi_Ei), label='参考曲线: $\\psi_{E}$')

# 指定中文字符字体
plt.xlabel('输入角 (度)', fontproperties=font_prop)
plt.ylabel('输出角 (度)', fontproperties=font_prop)
plt.title('曲柄摇杆机构输出角函数', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.grid(True)
plt.show()
