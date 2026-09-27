# Implementation: Compute Cosine similarity and Euclidean distance
import numpy as np
from scipy.spatial import distance

# Define vectors as given in the manual
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
B = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0])

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0])

# Compute Cosine Similarity (1 - Cosine Distance) and Euclidean Distance
cos_sim_AB = 1 - distance.cosine(A, B)
euc_dist_AB = distance.euclidean(A, B)

cos_sim_XY = 1 - distance.cosine(X, Y)
euc_dist_XY = distance.euclidean(X, Y)

# Print results in a formatted table
print(f"{'Measure':<20} | {'Vectors':<10} | {'Value'}")
print("-" * 45)
print(f"{'Cosine':<20} | {'(A, B)':<10} | {cos_sim_AB:<10.4f}")
print(f"{'Cosine':<20} | {'(X, Y)':<10} | {cos_sim_XY:<10.4f}")
print(f"{'Euclidean':<20} | {'(A, B)':<10} | {euc_dist_AB:<10.4f}")
print(f"{'Euclidean':<20} | {'(X, Y)':<10} | {euc_dist_XY:<10.4f}")