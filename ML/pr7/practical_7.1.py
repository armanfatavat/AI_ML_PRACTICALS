# Implementation: k-NN Classifier with 10-fold cross validation
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Load Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# Values of k to test
k_values = [1, 3, 5, 7]
accuracies = []

print(f"{'Value of k':<15} | {'Accuracy of model'}")
print("-" * 35)

# Use 10-fold cross validation and find accuracy for each k
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    # cv=10 performs 10-fold cross-validation
    scores = cross_val_score(knn, X, y, cv=10)
    mean_accuracy = scores.mean()
    accuracies.append(mean_accuracy)
    
    print(f"{k:<15} | {mean_accuracy:.4f}")