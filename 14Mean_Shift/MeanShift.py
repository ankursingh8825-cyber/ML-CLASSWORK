import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GIVEN DATA
# ----------------------------
X = np.array([[1],[2],[3],[10],[11],[12]])

# Bandwidth (window size)
bandwidth = 2

# Initial points (start from data points)
points = X.astype(float)

# ----------------------------
# MEAN SHIFT ALGORITHM
# ----------------------------
for _ in range(10):
    new_points = []

    for p in points:
        # Find points within bandwidth
        distances = np.abs(X - p)
        neighbors = X[distances <= bandwidth]

        # Compute mean of neighbors
        mean = np.mean(neighbors)
        new_points.append([mean])

    new_points = np.array(new_points)

    # Stop if points stop changing
    if np.allclose(points, new_points):
        break

    points = new_points

# Unique cluster centers
centers = np.unique(points.round(2), axis=0)

print("Cluster Centers:")
print(centers)

# ----------------------------
# PLOT RESULT
# ----------------------------
plt.figure()

plt.scatter(X, [0]*len(X), label="Data Points")
plt.scatter(centers, [0]*len(centers), marker='x', s=100, label="Cluster Centers")

plt.title("Mean Shift Clustering")
plt.xlabel("Data Points")
plt.legend()
plt.grid(True)

plt.show()