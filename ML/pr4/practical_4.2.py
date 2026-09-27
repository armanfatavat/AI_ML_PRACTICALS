# Set up diagram: Draw diagrams geometrically explaining cosine similarity and Euclidean distance
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define two simple 3D vectors to represent the geometry visually
origin = np.array([0, 0, 0])
vec_A = np.array([1.5, 0.5, 2])
vec_B = np.array([0.5, 2, 1.5])

# Draw vectors from origin
ax.quiver(*origin, *vec_A, color='blue', arrow_length_ratio=0.1)
ax.quiver(*origin, *vec_B, color='blue', arrow_length_ratio=0.1)

# Highlight endpoints A and B
ax.scatter(*vec_A, color='blue', s=80)
ax.text(vec_A[0], vec_A[1], vec_A[2] + 0.1, 'A', fontsize=12, fontweight='bold')

ax.scatter(*vec_B, color='blue', s=80)
ax.text(vec_B[0], vec_B[1], vec_B[2] + 0.1, 'B', fontsize=12, fontweight='bold')

# Draw Euclidean distance (dashed line between endpoints)
ax.plot([vec_A[0], vec_B[0]], [vec_A[1], vec_B[1]], [vec_A[2], vec_B[2]], 
        'gray', linestyle='dashed', label='dist(A, B)')

# Indicate Cosine similarity (Angle between vectors near origin)
ax.text(0.4, 0.4, 0.5, r'cos($\theta$)', color='black', fontsize=12, fontweight='bold')

# Set axes limits and labels to match the manual's diagram
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_zlim([0, 2.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Geometric Representation: Cosine Similarity and Euclidean Distance")
ax.legend()

plt.show()