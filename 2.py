import numpy as np
import matplotlib.pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# 定义微分方程的右侧
def f(x):
    return x

# 精确解
def exact_solution(x):
    return 2 * (np.exp(x) - np.exp(-x)) / (np.exp(1) - np.exp(-1)) - x

# 设置参数
h = 0.05
x0, x_end = 0, 1
n = int((x_end - x0) / h)

# 构造系数矩阵和右侧向量
A = np.zeros((n-1, n-1))
b = np.zeros(n-1)

# y[n-1] - (2+h**2)*y[n] + y[n+1] = x[n]*h**2
for i in range(n-1):
    x_i = (i + 1) * h
    b[i] = h**2 * f(x_i)
    if i == 0:
        A[i, i] = -(2 + h**2)
        A[i, i+1] = 1
    elif i == n-2:
        A[i, i-1] = 1
        A[i, i] = -(2 + h**2)
    else:
        A[i, i-1] = 1
        A[i, i] = -(2 + h**2)
        A[i, i+1] = 1

# 边界条件
b[0] -= 0
b[-1] -= 1

# np.linalg.solve(A, b)用于求解形如 
# 𝐴𝑦=𝑏
# 的线性方程组，其中 
# 𝐴 是一个系数矩阵，
# 𝑦 是未知数向量，
# 𝑏 是右侧的常数向量
y = np.linalg.solve(A, b)

# 加入边界条件的解
y = np.concatenate(([0], y, [1]))

# 计算精确解
xs = np.arange(x0, x_end + h, h)
ys_exact = exact_solution(xs)

# 输出结果表格
print("x      数值解       精确解       |数值解 - 精确解|")
for x, y_num, y_ex in zip(xs, y, ys_exact):
    print(f"{x:.2f}  {y_num:.6f}  {y_ex:.6f}  {abs(y_num - y_ex):.6e}")

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(xs, y, label='数值解', color='red', marker='o')
plt.plot(xs, ys_exact, label='精确解', color='black', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('差分方法解边值问题')
plt.grid(True)
plt.show()
