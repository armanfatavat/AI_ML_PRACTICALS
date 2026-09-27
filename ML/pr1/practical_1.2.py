import statistics

# Given Datasets from the manual
X1 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2]
X2 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5, 1, 2, 3, 4, 2, 5, 1, 3, 2, 1, 2, 1, 1, 1, 2000]
X3 = [1, 2, 3, 10, 20, 30, 100, 200, 300, 1000, 2000, 3000]

datasets = {"X1": X1, "X2": X2, "X3": X3}

# Print the formatted result table header
print(f"{'Dataset':<10} | {'Mean':<10} | {'Median':<10} | {'Mode'}")
print("-" * 55)

for name, data in datasets.items():
    mean_val = statistics.mean(data)
    median_val = statistics.median(data)
    
    # Use multimode to accurately capture bimodal data or data with no repeating values
    modes = statistics.multimode(data)
    
    # Check if all values are unique (meaning frequency is 1 for everything, like in X3)
    if len(modes) == len(data):
        mode_str = "No unique mode"
    else:
        # Join multiple modes if they exist (like in X1)
        mode_str = ", ".join(map(str, modes))
    
    print(f"{name:<10} | {mean_val:<10.2f} | {median_val:<10.2f} | {mode_str}")