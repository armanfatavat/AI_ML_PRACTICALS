import matplotlib.pyplot as plt
import statistics

# Given data for the setup diagram
X = [1, 3, 2, 4, 5, 6, 4, 3, 2, 4, 5, 3, 1, 2, 3, 2, 3, 1, 4]

# Calculate statistical measures
mean_val = statistics.mean(X)
median_val = statistics.median(X)
mode_val = statistics.mode(X)

# Plot the histogram
plt.figure(figsize=(9, 6))
plt.hist(X, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], color="#4A90E2", edgecolor="black", rwidth=0.85)

# Add vertical lines for Mean, Median, and Mode
plt.axvline(mean_val, color="red", linestyle="--", linewidth=2.5, label=f"Mean: {mean_val:.2f}")
plt.axvline(median_val, color="lightgreen", linestyle="-", linewidth=2.5, label=f"Median: {median_val}")
plt.axvline(mode_val, color="orange", linestyle=":", linewidth=3, label=f"Mode: {mode_val}")

# Formatting the plot
plt.title("Histogram of Data X", fontsize=14, fontweight="bold")
plt.xlabel("Values", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()