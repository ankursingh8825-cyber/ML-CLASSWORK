import math
import matplotlib.pyplot as plt
from collections import Counter

height = [158,158,158,160,160,163,163,160,163,165,165,165,168,168,168,170,170]
weight = [58,59,63,59,60,60,61,64,64,61,62,65,62,66,63,64,68]
tshirt = ["M","M","M","M","M","M","M","M","L","L","L","L","L","L","L","L","L"]

test_height = 161
test_weight = 61

data = list(zip(height, weight, tshirt))

distances = []
for h, w, size in data:
    d = math.sqrt((h - test_height)**2 + (w - test_weight)**2)
    distances.append((d, size))

distances.sort(key=lambda x: x[0])

k_values = [1, 3, 5, 7, 9, 11, 13, 15]

print("Predictions:")
for k in k_values:
    labels = [label for _, label in distances[:k]]
    prediction = Counter(labels).most_common(1)[0][0]
    print(f"k = {k} -> {prediction}")

for i in range(len(height)):
    if tshirt[i] == "M":
        plt.scatter(height[i], weight[i], marker='o')
    elif tshirt[i] == "ML":
        plt.scatter(height[i], weight[i], marker='s')
    else:
        plt.scatter(height[i], weight[i], marker='^')

plt.scatter(test_height, test_weight, marker='*', s=200)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("KNN Classification (Height vs Weight)")
plt.show()
