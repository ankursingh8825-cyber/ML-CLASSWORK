import pandas as pd
import numpy as np
from collections import Counter
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# LOAD DATASET
# ===============================
df = pd.read_csv("Iris.csv")

# Remove ID if present
if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

# Encode species into numeric codes
df["Species_code"] = df["Species"].astype("category").cat.codes

# ===============================
# MANUAL KNN (Without sklearn)
# ===============================
df = df.sample(frac=1).reset_index(drop=True)  # shuffle dataset
split_index = int(0.8 * len(df))
train = df.iloc[:split_index]
test = df.iloc[split_index:]


def euclidean(x1, x2):
    return sqrt(np.sum((x1 - x2) ** 2))


def knn_predict(test_row, train, k):
    distances = []
    for _, train_row in train.iterrows():
        dist = euclidean(test_row.iloc[:-2], train_row.iloc[:-2])  # ignore last 2 columns
        distances.append((dist, train_row["Species_code"]))

    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    labels = [lbl for _, lbl in k_neighbors]

    return Counter(labels).most_common(1)[0][0]


k = 3
correct = 0
num_classes = df["Species_code"].nunique()
conf_matrix_manual = np.zeros((num_classes, num_classes), dtype=int)

for _, row in test.iterrows():
    actual = row["Species_code"]
    pred = knn_predict(row, train, k)

    if actual == pred:
        correct += 1

    conf_matrix_manual[actual][pred] += 1

manual_accuracy = (correct / len(test)) * 100

# ===============================
# KNN WITH SKLEARN
# ===============================
X = df.iloc[:, :-2]  # remove Species & Species_code
y = df["Species_code"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
library_accuracy = accuracy_score(y_test, y_pred) * 100

# ===============================
# PRINT RESULTS
# ===============================
print("Manual KNN Accuracy      :", round(manual_accuracy, 2), "%")
print("\nManual Confusion Matrix:")
print(conf_matrix_manual)

print("\nSklearn KNN Accuracy     :", round(library_accuracy, 2), "%")

# ===============================
# PLOT MANUAL CONFUSION MATRIX (Heatmap)
# ===============================
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_manual, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Manual KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===============================
# SCATTER PLOT OF IRIS DATASET
# ===============================
plt.figure(figsize=(7, 5))
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c=df["Species_code"], cmap="viridis")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Dataset Visualization (Colored by Species)")
plt.colorbar(label="Species Code")
plt.show()
