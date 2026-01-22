import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GIVEN DATA
# ----------------------------
x = np.array([-2, -1, 0, 1, 2])
y = np.array([1, 2, 3, 3, 4])

n = len(y)

# ----------------------------
# DESIGN MATRIX (with intercept)
# ----------------------------
X = np.column_stack((np.ones(n), x))

# ----------------------------
# NORMAL EQUATION
# β = (XᵀX)⁻¹ Xᵀy
# ----------------------------
beta = np.linalg.inv(X.T @ X) @ X.T @ y

b0, b1 = beta

# ----------------------------
# PRINT REGRESSION EQUATION
# ----------------------------
print("\n--- Linear Regression Equation ---")
print(f"y = {b0:.4f} + {b1:.4f}x")

# ----------------------------
# PREDICTED VALUES
# ----------------------------
y_pred = b0 + b1 * x

# ----------------------------
# PLOT DATA + REGRESSION LINE
# ----------------------------
plt.figure()
plt.scatter(x, y, color="red", label="Data Points")
plt.plot(x, y_pred, color="blue", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
