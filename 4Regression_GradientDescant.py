import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# DATA + INITIAL PARAMETERS
# ----------------------------
x = np.array([1,2,3,4,5])
y = np.array([6,3,4,4,6])

b0 = 1.0
b1 = 1.0
alpha = 0.1
n_iter = 5
n = len(x)

params = [(b0, b1)]

# ----------------------------
# Compute gradients and loss
# ----------------------------
def compute_values(b0, b1):
    e = (b0 + b1 * x) - y
    sum_e = np.sum(e)
    sum_ex = np.sum(e * x)
    sse = np.sum(e**2)
    return sum_e, sum_ex, sse

# ----------------------------
# Gradient Descent
# ----------------------------
print("\n--- Line Equation After Each Iteration ---")
print(f"Iteration 0: y = {b0:.4f} + {b1:.4f}x")

for it in range(1, n_iter + 1):
    sum_e, sum_ex, sse = compute_values(b0, b1)

    b0 -= alpha * (sum_e / n)
    b1 -= alpha * (sum_ex / n)

    params.append((b0, b1))

    # ✅ PRINT EQUATION
    print(f"Iteration {it}: y = {b0:.4f} + {b1:.4f}x")

# ----------------------------
# SINGLE FIGURE (2D + 3D)
# ----------------------------
fig = plt.figure(figsize=(16, 6))

# -------- 2D: Line Updates --------
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(x, y, color="orange", s=80, marker="x", label="Data")

x_line = np.linspace(min(x), max(x), 200)
for i, (b0p, b1p) in enumerate(params):
    ax1.plot(x_line, b0p + b1p * x_line, label=f"Iter {i}")

ax1.set_title("Regression Line Updates (Gradient Descent)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True)

# -------- 3D: Loss Surface + Path --------
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

b0_vals = np.linspace(-2, 4, 50)
b1_vals = np.linspace(-2, 4, 50)
B0, B1 = np.meshgrid(b0_vals, b1_vals)

SSE = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        _, _, sse = compute_values(B0[i,j], B1[i,j])
        SSE[i,j] = sse

ax2.plot_surface(B0, B1, SSE, cmap="viridis", alpha=0.7)

path_b0 = [p[0] for p in params]
path_b1 = [p[1] for p in params]
path_sse = [compute_values(p[0], p[1])[2] for p in params]

ax2.plot(path_b0, path_b1, path_sse,
         color="red", marker="o", linewidth=3, label="GD Path")

ax2.set_title("3D Loss Surface & Gradient Descent Path")
ax2.set_xlabel("b0")
ax2.set_ylabel("b1")
ax2.set_zlabel("SSE")
ax2.legend()

plt.show()
