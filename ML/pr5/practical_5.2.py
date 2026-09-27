# Plot: Plot the given data, fit the regression line, and show the predicted value for X = 210
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Data and model fitting
X = np.array([70, 80, 90, 100, 110, 120, 130, 140, 150, 160]).reshape(-1, 1)
Y = np.array([7, 7, 8, 9, 12, 12, 15, 14, 13, 17])

model = LinearRegression()
model.fit(X, Y)

# Generate points for the regression line
X_line = np.linspace(60, 220, 100).reshape(-1, 1)
Y_line = model.predict(X_line)

# Prediction point
X_pred = 210
Y_pred = model.predict(np.array([[X_pred]]))[0]

# Create the plot
plt.figure(figsize=(10, 6))

# Scatter plot of the original dataset
plt.scatter(X, Y, color='blue', label='Actual Data (Training)', s=50)

# Plot the regression line
plt.plot(X_line, Y_line, color='red', label=f'Regression Line ($Y = {model.intercept_:.2f} + {model.coef_[0]:.2f}X$)', linewidth=2)

# Highlight the predicted value
plt.scatter([X_pred], [Y_pred], color='green', marker='*', s=200, label=f'Predicted Value at X=210 (Y={Y_pred:.2f})')

# Draw lines to highlight the predicted point on axes
plt.axvline(x=X_pred, color='gray', linestyle='--', alpha=0.6)
plt.axhline(y=Y_pred, color='gray', linestyle='--', alpha=0.6)

# Formatting
plt.title("Simple Linear Regression: Advertisement Spend vs Unit Sale Increase")
plt.xlabel("Amount Spent for Advertisement (X)")
plt.ylabel("Increase in Unit Sale (Y)")
plt.legend(loc='upper left')
plt.grid(alpha=0.4)

plt.show()git switch dev
git switch -c feature/ML-pr5