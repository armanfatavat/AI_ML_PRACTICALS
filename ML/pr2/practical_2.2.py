import numpy as np

# Implementation: Compute Standard Deviation and Variance
X1 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2]
X2 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2000]
X3 = [1, 2, 3, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000]

datasets = {"X1": X1, "X2": X2, "X3": X3}

print(f"{'Dataset':<10} | {'Standard Deviation':<20} | {'Variance'}")
print("-" * 50)

for name, data in datasets.items():
    # ddof=0 is used to match the population formula in the manual
    std_dev = np.std(data, ddof=0)
    variance = np.var(data, ddof=0)
    
    print(f"{name:<10} | {std_dev:<20.2f} | {variance:.2f}")