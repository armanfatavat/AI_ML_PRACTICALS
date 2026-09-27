# Plot: Line chart for k versus accuracy
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load dataset and compute accuracies
iris = load_iris()
X = iris.data
y = iris.target

k_values = [1, 3, 5, 7]
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10)
    accuracies.append(scores.mean())

# Plot the line chart
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)

# Formatting the plot
plt.title('k-NN Classifier Accuracy vs. Value of k (10-fold CV)', fontsize=14, fontweight='bold')
plt.xlabel('Value of k (Number of Neighbors)', fontsize=12)
plt.ylabel('Mean Accuracy', fontsize=12)
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()