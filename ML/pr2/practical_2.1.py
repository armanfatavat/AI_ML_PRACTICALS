import matplotlib.pyplot as plt
import numpy as np

# Set up diagram: Histogram Plotting
X = [1, 3, 2, 4, 56, 4, 3, 2, 4, 5, 3, 1, 2, 3, 2, 3, 1, 4]

mean_val = np.mean(X)
std_val = np.std(X, ddof=0)
var_val = np.var(X, ddof=0)

plt.figure(figsize=(10, 6))
plt.hist(X, bins=30, color="skyblue", edgecolor="black")

plt.axvline(mean_val, color="red", linestyle="dashed", linewidth=2, label=f"Mean: {mean_val:.2f}")
plt.axvline(mean_val + std_val, color="green", linestyle="dotted", linewidth=2, label=f"+1 Std Dev: {mean_val + std_val:.2f}")
plt.axvline(mean_val - std_val, color="green", linestyle="dotted", linewidth=2, label=f"-1 Std Dev: {mean_val - std_val:.2f}")

plt.text(0.5, 0.85, f"Standard Deviation: {std_val:.2f}\nVariance: {var_val:.2f}", 
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

plt.title("Histogram of Data X")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.legend()
plt.show()