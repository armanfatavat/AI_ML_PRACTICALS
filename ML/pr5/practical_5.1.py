# Implementation: Simple Linear Regression Model and Predictions
import numpy as np
from sklearn.linear_model import LinearRegression

# Given data for Advertisement Spend (X) and Unit Sale Increase (Y)
X = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160]).reshape(-1, 1)
Y = np.array([7, 7, 8, 9, 12, 12, 15, 14, 13, 17])

# Create and fit the simple linear regression model
model = LinearRegression()
model.fit(X, Y)

# Find model parameters w0 (intercept) and w1 (slope)
w0 = model.intercept_
w1 = model.coef_[0]

# Predict Y for X = 210
X_test = np.array([[210]])
Y_pred = model.predict(X_test)[0]

# Print results in a formatted table
print(f"{'Variable':<15} | {'Value'}")
print("-" * 30)
print(f"{'w0 (Intercept)':<15} | {w0:.4f}")
print(f"{'w1 (Slope)':<15} | {w1:.4f}")
print(f"{'Y | X=210':<15} | {Y_pred:.4f}")