import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GIVEN DATA
# ----------------------------
X = np.array([
    [2,3],
    [3,4],
    [4,5],
    [5,6],
    [6,7]
])

# ----------------------------
# STEP 1: MEAN CENTERING
# ----------------------------
mean = np.mean(X, axis=0)
X_centered = X - mean

# ----------------------------
# STEP 2: SVD DECOMPOSITION
# X = U Σ Vᵀ
# ----------------------------
U, S, VT = np.linalg.svd(X_centered)

# ----------------------------
# STEP 3: SELECT k COMPONENTS
# ----------------------------
k = 1
V_k = VT[:k].T

# ----------------------------
# STEP 4: PROJECT DATA
# ----------------------------
X_reduced = X_centered @ V_k

print("Reduced Data:\n", X_reduced)

# ----------------------------
# STEP 5: RECONSTRUCT DATA
# ----------------------------
X_reconstructed = X_reduced @ V_k.T + mean

# ----------------------------
# PLOT DATA
# ----------------------------
plt.figure()

plt.scatter(X[:,0], X[:,1], label="Original Data")
plt.scatter(X_reconstructed[:,0], X_reconstructed[:,1], marker='x', label="Reconstructed Data")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("PCA using SVD")
plt.legend()
plt.grid(True)

plt.show()