import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GIVEN DATA
# ----------------------------
X = np.array([[1], [2], [3], [10], [11], [12]])

# Number of clusters
k = 2

# Initial centroids
centroids = np.array([[2], [11]])

# ----------------------------
# K-MEANS ALGORITHM
# ----------------------------
for _ in range(10):

    # Calculate distance from centroids
    distances = np.abs(X - centroids.T)

    # Assign clusters
    clusters = np.argmin(distances, axis=1)

    # Update centroids
    new_centroids = np.array([X[clusters == i].mean() for i in range(k)]).reshape(k, 1)

    # Stop if centroids do not change
    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

print("Final Centroids:")
print(centroids)

print("\nCluster Assignments:")
print(clusters)

# ----------------------------
# PLOT RESULT
# ----------------------------
plt.figure()

for i in range(k):
    plt.scatter(X[clusters == i], [0] * len(X[clusters == i]), label=f"Cluster {i + 1}")

plt.scatter(centroids, [0] * k, marker='x', s=100, label="Centroids")

plt.title("K-Means Clustering")
plt.xlabel("Data Points")
plt.legend()
plt.grid(True)

plt.show()