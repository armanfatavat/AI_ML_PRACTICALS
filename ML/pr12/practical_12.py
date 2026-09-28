# Implementation: Demonstrate use of NumPy, Pandas, Scikit-learn, and TensorFlow
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import tensorflow as tf

print("=========================================")
print(" 1. NumPy Demonstration (Numerical Math) ")
print("=========================================")
# NumPy is used for efficient mathematical operations on arrays
arr = np.array([1, 2, 3, 4, 5])
arr_squared = np.square(arr)
print(f"Original Array: {arr}")
print(f"Squared Array:  {arr_squared}")

print("\n=========================================")
print(" 2. Pandas Demonstration (Data Handling) ")
print("=========================================")
# Pandas is used for cleaning and organizing tabular data
data = {'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
        'Age': [25, 30, 25, 35]}
df = pd.DataFrame(data)
print("Original DataFrame (contains duplicate):")
print(df)
df_clean = df.drop_duplicates()
print("\nDataFrame after df.drop_duplicates():")
print(df_clean)

print("\n=========================================")
print(" 3. Scikit-learn Demonstration (ML)      ")
print("=========================================")
# Scikit-learn is used for preprocessing and classical ML algorithms
X = np.array([[10], [20], [30], [40]])
y = np.array([1, 2, 3, 4])

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Standardized Features:\n{X_scaled}")

# Modeling
model = LinearRegression().fit(X_scaled, y)
print(f"Linear Regression R^2 Score: {model.score(X_scaled, y):.2f}")

print("\n=========================================")
print(" 4. TensorFlow Demonstration (Deep Learn)")
print("=========================================")
# TensorFlow is used for deep learning and neural networks via tensors
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[2, 0], [0, 2]])
tensor_mult = tf.matmul(tensor_a, tensor_b)

print("Tensor A:")
print(tensor_a.numpy())
print("\nMatrix Multiplication (Tensor A * Tensor B):")
print(tensor_mult.numpy())