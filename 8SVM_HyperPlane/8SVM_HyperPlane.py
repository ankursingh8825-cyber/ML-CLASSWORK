# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Generate sample 2D dataset
X, y = datasets.make_blobs(n_samples=100, centers=2,
                           random_state=6, cluster_std=1.2)

# Create SVM model with linear kernel
model = SVC(kernel='linear')
model.fit(X, y)

# Get hyperplane parameters
w = model.coef_[0]      # weights
b = model.intercept_[0] # bias

# Create grid for plotting
x_points = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100)
y_points = -(w[0] * x_points + b) / w[1]  # Decision boundary

# Margin lines
margin = 1 / np.sqrt(np.sum(w**2))
y_margin_positive = y_points + np.sqrt(1 + (w[0]/w[1])**2) * margin
y_margin_negative = y_points - np.sqrt(1 + (w[0]/w[1])**2) * margin

# Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

# Plot hyperplane
plt.plot(x_points, y_points, 'k-', label='Hyperplane')

# Plot margins
plt.plot(x_points, y_margin_positive, 'k--')
plt.plot(x_points, y_margin_negative, 'k--')

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k',
            label='Support Vectors')

plt.title("SVM Maximum Margin Hyperplane")
plt.legend()
plt.show()
