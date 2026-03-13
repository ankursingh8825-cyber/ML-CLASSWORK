import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# GIVEN DATA
# ----------------------------
X = np.array([[1],[2],[3],[10],[11],[12]])

# Number of clusters
k = 2

# Initial medians
medians = np.array([[2],[11]])

# ----------------------------
# K-MEDIAN ALGORITHM
# ----------------------------
for _ in range(10):

    # Compute Manhattan distance
    distances = np.abs(X - medians.T)

    # Assign clusters
    clusters = np.argmin(distances, axis=1)

    # Update medians
    new_medians = np.array([np.median(X[clusters == i]) for i in range(k)]).reshape(k,1)

    # Stop if medians do not change
    if np.all(medians == new_medians):
        break

    medians = new_medians

print("Final Medians:")
print(medians)

print("\nCluster Assignments:")
print(clusters)

# ----------------------------
# PLOT RESULT
# ----------------------------
plt.figure()

for i in range(k):
    plt.scatter(X[clusters==i], [0]*len(X[clusters==i]), label=f"Cluster {i+1}")

plt.scatter(medians, [0]*k, marker='x', s=100, label="Medians")

plt.title("K-Median Clustering")
plt.xlabel("Data Points")
plt.legend()
plt.grid(True)

plt.show()