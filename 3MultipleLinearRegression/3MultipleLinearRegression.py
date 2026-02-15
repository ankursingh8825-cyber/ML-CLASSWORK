import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.array([3, 4, 5, 6, 2])
x2 = np.array([8, 5, 7, 3, 1])
y  = np.array([-3.7, 3.5, 2.5, 11.5, 5.7])

n = len(y)

# ----------------------------
# DESIGN MATRIX (with intercept)
# ----------------------------
X = np.column_stack((np.ones(n), x1, x2))

# ----------------------------
# NORMAL EQUATION
# β = (XᵀX)⁻¹ Xᵀy
# ----------------------------
beta = np.linalg.inv(X.T @ X) @ X.T @ y

b0, b1, b2 = beta

# ----------------------------
# PRINT REGRESSION EQUATION
# ----------------------------
print("\n--- Multiple Linear Regression Equation ---")
print(f"y = {b0:.4f} + {b1:.4f}*x1 + {b2:.4f}*x2")

# ----------------------------
# PREDICTED VALUES
# ----------------------------
y_pred = b0 + b1 * x1 + b2 * x2

# ----------------------------
# 2D: ACTUAL vs PREDICTED
# ----------------------------
plt.figure()
plt.scatter(y, y_pred, color="blue")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()

# ----------------------------
# 3D: DATA + REGRESSION PLANE
# ----------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x1, x2, y, color="red", s=60, label="Data Points")

x1_surf, x2_surf = np.meshgrid(
    np.linspace(min(x1), max(x1), 20),
    np.linspace(min(x2), max(x2), 20)
)

y_surf = b0 + b1 * x1_surf + b2 * x2_surf

ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.5)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Multiple Linear Regression Plane")
ax.legend()

plt.show()
