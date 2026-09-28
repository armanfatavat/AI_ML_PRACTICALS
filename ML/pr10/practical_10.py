import numpy as np
from sklearn.datasets import load_iris

# 1. Load Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# Overall mean (m) of the entire dataset
m = np.mean(X, axis=0)
num_features = X.shape[1]

# Initialize the scatter matrices (4x4 matrices for Iris's 4 features)
S_W = np.zeros((num_features, num_features))
S_B = np.zeros((num_features, num_features))
S_T = np.zeros((num_features, num_features))

# 2. Calculate Total Scatter (S_T)
for x in X:
    x_minus_m = (x - m).reshape(num_features, 1)
    S_T += x_minus_m.dot(x_minus_m.T)

# 3. Calculate Within-class (S_W) and Between-class (S_B) scatter
classes = np.unique(y)
for c in classes:
    # Filter data for the specific class
    X_c = X[y == c]
    
    # Class mean (m_i)
    m_i = np.mean(X_c, axis=0)
    
    # Number of samples in class (n_i)
    n_i = X_c.shape[0]
    
    # Within-class scatter for this specific class
    S_i = np.zeros((num_features, num_features))
    for x in X_c:
        x_minus_m_i = (x - m_i).reshape(num_features, 1)
        S_i += x_minus_m_i.dot(x_minus_m_i.T)
    S_W += S_i
    
    # Between-class scatter for this specific class
    m_i_minus_m = (m_i - m).reshape(num_features, 1)
    S_B += n_i * (m_i_minus_m).dot(m_i_minus_m.T)

# To provide a single scalar "Value" for the table as requested by the manual,
# we compute the trace (sum of the diagonal elements) of the matrices.
# This represents the sum of variances (squared distances) across all features.
value_SW = np.trace(S_W)
value_SB = np.trace(S_B)
value_ST = np.trace(S_T)

# Print Results in the manual's requested format
print(f"{'Measure':<25} | {'Value'}")
print("-" * 35)
print(f"{'Within class scatter':<25} | {value_SW:.4f}")
print(f"{'Between class scatter':<25} | {value_SB:.4f}")
print(f"{'Total Scatter':<25} | {value_ST:.4f}")