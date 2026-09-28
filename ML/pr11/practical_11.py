import numpy as np
from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Divide dataset into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Helper function to map unsupervised cluster labels to true labels to calculate "accuracy"
def evaluate_kmeans_accuracy(X_train, y_train, X_test, y_test, k, normalize=False):
    # If simulating Cosine distance, normalize the data (Spherical K-Means)
    if normalize:
        scaler = Normalizer()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    # Initialize and train K-Means
    # n_init=10 explicitly defined to suppress warnings
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    
    # Map each cluster to the most frequent actual class label in the training set
    train_clusters = kmeans.predict(X_train)
    cluster_to_label_map = {}
    
    for i in range(k):
        # Find true labels for data points assigned to cluster i
        true_labels_in_cluster = y_train[train_clusters == i]
        if len(true_labels_in_cluster) > 0:
            # Assign the most frequent true label to this cluster
            most_frequent_label = mode(true_labels_in_cluster, keepdims=True).mode[0]
            cluster_to_label_map[i] = most_frequent_label
        else:
            cluster_to_label_map[i] = -1 # Empty cluster fallback
            
    # Predict clusters for the 30% test data
    test_clusters = kmeans.predict(X_test)
    
    # Convert test cluster assignments to actual class labels using our map
    predicted_labels = [cluster_to_label_map[c] for c in test_clusters]
    
    # Calculate and return accuracy
    return accuracy_score(y_test, predicted_labels)

# 3. Define parameter combinations for K and Distance Measure
# Note: standard scikit-learn KMeans uses Euclidean distance. 
# We simulate Cosine distance by L2-normalizing the data before applying KMeans.
experiments = [
    {"k": 2, "distance": "Euclidean", "normalize": False},
    {"k": 3, "distance": "Euclidean", "normalize": False},
    {"k": 4, "distance": "Euclidean", "normalize": False},
    {"k": 5, "distance": "Euclidean", "normalize": False},
    {"k": 3, "distance": "Cosine (Normalized)", "normalize": True},
]

# Print Results in the manual's requested format
print(f"{'Value of K':<15} | {'Distance measure':<25} | {'Accuracy'}")
print("-" * 55)

for exp in experiments:
    acc = evaluate_kmeans_accuracy(
        X_train, y_train, X_test, y_test, 
        k=exp["k"], 
        normalize=exp["normalize"]
    )
    print(f"{exp['k']:<15} | {exp['distance']:<25} | {acc:.4f}")