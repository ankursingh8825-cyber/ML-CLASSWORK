import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# ----------------------------
# GIVEN DATA
# ----------------------------
X = np.array([[1],[2],[3],[10],[11],[12]])

# ----------------------------
# AGGLOMERATIVE CLUSTERING
# ----------------------------

# Single Linkage
Z_single = linkage(X, method='single')

plt.figure()
dendrogram(Z_single)
plt.title("Agglomerative Clustering - Single Linkage")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# Complete Linkage
Z_complete = linkage(X, method='complete')

plt.figure()
dendrogram(Z_complete)
plt.title("Agglomerative Clustering - Complete Linkage")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# Average Linkage
Z_average = linkage(X, method='average')

plt.figure()
dendrogram(Z_average)
plt.title("Agglomerative Clustering - Average Linkage")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# Centroid Linkage
Z_centroid = linkage(X, method='centroid')

plt.figure()
dendrogram(Z_centroid)
plt.title("Agglomerative Clustering - Centroid Linkage")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()


# ----------------------------
# DIVISIVE HIERARCHICAL CLUSTERING (Simple Simulation)
# ----------------------------

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(X)

plt.figure()

for i in range(2):
    plt.scatter(X[labels==i], [0]*len(X[labels==i]), label=f"Cluster {i+1}")

plt.title("Divisive Hierarchical Clustering (Top-down split)")
plt.xlabel("Data Points")
plt.legend()
plt.grid(True)

plt.show()