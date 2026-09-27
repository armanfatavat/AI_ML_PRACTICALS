# Set up diagram: Plot all three classes in dataset with 2 dimensions only
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load and process data (carrying over from implementation)
iris = load_iris()
X_scaled = StandardScaler().fit_transform(iris.data)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

y = iris.target
target_names = iris.target_names

# Plotting the 2D data
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], 
                color=color, alpha=0.8, s=50, label=target_name)

plt.title('PCA of IRIS Dataset (2 Dimensions)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='best', title="Flower Classes")
plt.grid(alpha=0.3)
plt.show()