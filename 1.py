import numpy as np
import matplotlib.pyplot as plt
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 定义微分方程
def f(x, y):
    return x + y

# 精确解
def exact_solution(x):
    return 2 * np.exp(x) - x - 1

# 改进欧拉法 (Heun 法)
def improved_euler_method(f, y0, x0, x_end, h):
    xs = np.arange(x0, x_end + h, h)
    ys = np.zeros(xs.shape)
    ys[0] = y0

    for i in range(1, len(xs)):
        x = xs[i - 1]
        y = ys[i - 1]
        k1 = f(x, y)
        k2 = f(x + h, y + h * k1)
        ys[i] = y + (h / 2) * (k1 + k2)

    return xs, ys

# 四阶龙格-库塔法 (RK4)
def runge_kutta_method(f, y0, x0, x_end, h):
    xs = np.arange(x0, x_end + h, h)
    ys = np.zeros(xs.shape)
    ys[0] = y0

    for i in range(1, len(xs)):
        x = xs[i - 1]
        y = ys[i - 1]
        # 直接计算 K1*h,便于使用
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)
        ys[i] = y + (k1 + 2*k2 + 2*k3 + k4) / 6

    return xs, ys

# 参数
y0 = 1
x0 = 0
x_end = 1
h = 0.02

# 计算数值解
xs_improved_euler, ys_improved_euler = improved_euler_method(f, y0, x0, x_end, h)
xs_runge_kutta, ys_runge_kutta = runge_kutta_method(f, y0, x0, x_end, h)

# 计算精确解
xs_exact = np.arange(x0, x_end + h, h)
ys_exact = exact_solution(xs_exact)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(xs_exact, ys_exact, label='精确解', color='green', linestyle='--')
plt.plot(xs_improved_euler, ys_improved_euler, label='改进欧拉法', color='red',linestyle='-')
plt.plot(xs_runge_kutta, ys_runge_kutta, label='四阶龙格-库塔法', color='blue',linestyle='dotted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('微分方程数值解法')
plt.grid(True)
plt.show()

# 显示结果表格
print("x   精确解    改进欧拉法  四阶龙格-库塔法    改进欧拉法误差    四阶龙格-库塔法误差")
for x, y_exact, y_euler, y_rk4, y_euler_exact, y_rk4_exact in zip(xs_exact, ys_exact, ys_improved_euler, ys_runge_kutta, ys_improved_euler - ys_exact, ys_runge_kutta - ys_exact):
    print(f"{x:.2f}  {y_exact:.6f}  {y_euler:.6f}  {y_rk4:.6f}  {y_euler_exact:.6f}  {y_rk4_exact:.10f}")
