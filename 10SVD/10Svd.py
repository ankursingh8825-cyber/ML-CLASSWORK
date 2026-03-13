import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Define Matrix
# -----------------------------
A = np.array([[4, 8],
              [8, 4]])

print("Original Matrix A:\n", A)


# -----------------------------
# Step 2: Compute SVD
# -----------------------------
U, Sigma, VT = np.linalg.svd(A)

# Convert Sigma vector to diagonal matrix
Sigma_matrix = np.zeros((2, 2))
Sigma_matrix[0, 0] = Sigma[0]
Sigma_matrix[1, 1] = Sigma[1]

print("\nMatrix U:\n", U)
print("\nSigma Matrix:\n", Sigma_matrix)
print("\nMatrix V^T:\n", VT)


# -----------------------------
# Step 3: Reconstruct Matrix
# -----------------------------
A_reconstructed = U @ Sigma_matrix @ VT

print("\nReconstructed Matrix:\n", A_reconstructed)


# -----------------------------
# Step 4: Visualization 1
# Original Matrix Heatmap
# -----------------------------
plt.figure()
plt.imshow(A)
plt.title("Original Matrix A")
plt.colorbar()
plt.show()


# -----------------------------
# Step 5: Visualization 2
# Reconstructed Matrix Heatmap
# -----------------------------
plt.figure()
plt.imshow(A_reconstructed)
plt.title("Reconstructed Matrix from SVD")
plt.colorbar()
plt.show()


# -----------------------------
# Step 6: Geometric Visualization
# Unit Circle Transformation
# -----------------------------
theta = np.linspace(0, 2*np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])

# Transform circle using A
transformed = A @ circle

plt.figure()
plt.plot(circle[0], circle[1])
plt.title("Unit Circle")
plt.gca().set_aspect('equal')
plt.show()

plt.figure()
plt.plot(transformed[0], transformed[1])
plt.title("Transformed by Matrix A (Ellipse)")
plt.gca().set_aspect('equal')
plt.show()