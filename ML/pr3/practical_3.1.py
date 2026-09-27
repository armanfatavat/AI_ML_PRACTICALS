# Implementation: Load Iris dataset and perform dimension reduction using PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

# 1. Load Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Standardize the data (as required by PCA theory)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Reduce the dimension of dataset to 2 by applying PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Display the shapes to confirm dimension reduction
print(f"Original dataset shape: {X.shape}")
print(f"Reduced dataset shape (after PCA): {X_pca.shape}")

# Optional: View the first few rows of the reduced data
df_pca = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
df_pca['Target'] = y
print("\nFirst 5 rows of reduced dataset:\n", df_pca.head())