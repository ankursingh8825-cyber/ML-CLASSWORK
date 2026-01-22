# Multiple Linear Regression (x1, x2 → y)
# Using Normal Equation (No sklearn) + Visualization

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# Step 1: Given data (already fixed)
# ===============================
x1 = [1,2,3]
x2 = [3,4,7]
y  = [2,5,9]

n = len(x1)

# ===============================
# Step 2: Convert to numpy arrays
# ===============================
X1 = np.array(x1)
X2 = np.array(x2)
Y  = np.array(y)

# ===============================
# Step 3: Design matrix (with intercept)
# ===============================
X = np.column_stack((np.ones(n), X1, X2))

# ===============================
# Step 4: Normal Equation
# β = (XᵀX)⁻¹ XᵀY
# ===============================
beta = np.linalg.inv(X.T @ X) @ X.T @ Y

b0, b1, b2 = beta

# ===============================
# Step 5: Print regression equation
# ===============================
print("\n--- Regression Equation ---")
print(f"y = {b0:.4f} + {b1:.4f}*x1 + {b2:.4f}*x2")

# ===============================
# Step 6: Predicted Y values
# ===============================
Y_pred = b0 + b1 * X1 + b2 * X2

# ===============================
# Visualization 1: Actual vs Predicted
# ===============================
plt.figure()
plt.scatter(Y, Y_pred)
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Actual vs Predicted Values")
plt.grid(True)
plt.show()

# ===============================
# Visualization 2: 3D Scatter + Plane
# ===============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, Y)

x1_surf, x2_surf = np.meshgrid(
    np.linspace(X1.min(), X1.max(), 20),
    np.linspace(X2.min(), X2.max(), 20)
)

y_surf = b0 + b1 * x1_surf + b2 * x2_surf

ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.5)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")
ax.set_title("3D Multiple Linear Regression Plane")

plt.show()
