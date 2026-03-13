import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
# APPLY PCA
# ----------------------------
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

print("Original Data:\n", X)
print("\nPrincipal Component:\n", X_pca)

# ----------------------------
# PLOT ORIGINAL DATA
# ----------------------------
plt.figure()
plt.scatter(X[:,0], X[:,1], label="Original Data")

# PCA direction
mean = np.mean(X, axis=0)
vector = pca.components_[0]

plt.quiver(mean[0], mean[1], vector[0], vector[1],
           angles='xy', scale_units='xy', scale=1, label="PC1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Principal Component Analysis")
plt.legend()
plt.grid(True)
plt.show()