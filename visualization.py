import random
import matplotlib.pyplot as plt
from math import sqrt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

import grad
from parameters import Parameters
from tqdm import tqdm
import numpy as np
import csv


# Function definition for the 2D Rosenbrock function
def f_r_2(x):
    return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0] * x[1]) - (4 * x[0]) - (4 * x[1]) + 1


# Function to read data from CSV
def read_csv(filename):
    points = []
    distances = []
    iterations = []
    function_evals = []

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            x, y, distance, iteration, func_eval = map(float, row)
            points.append([x, y])
            distances.append(distance)
            iterations.append(iteration)
            function_evals.append(func_eval)

    return np.array(points), np.array(distances), np.array(iterations), np.array(function_evals)


f1_min = [1.13263815658242, -0.46597244636103796]
f2_min = [1, 1]
delta = 0.1

x_range = np.linspace(f2_min[0] - delta, f2_min[0] + delta, 400)
y_range = np.linspace(f2_min[1] - delta, f2_min[1] + delta, 400)

X, Y = np.meshgrid(x_range, y_range)
Z = f_r_2([X, Y])

# Plotting the function
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Marking the minimum point
ax.scatter(f2_min[0], f2_min[1], f1(f1_min), color='red', marker='x', s=100)

ax.set_title('3D surface plot of the function f2')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f1(x)')

plt.show()


# Read data from CSV
points, distance, its, function_evals = read_csv('f2_replace_oldest_1e-5_tol_1e-2_delta_1e-1_range_1000.csv')

# Plot 1: x-coordinate vs. y-coordinate with number of iterations as color bar
x_coords = points[:, 0]
y_coords = points[:, 1]

plt.figure(figsize=(10, 6))
sc = plt.scatter(x_coords, y_coords, c=its, cmap='viridis', alpha=0.75, edgecolor='k', s=100)
plt.colorbar(sc, label='Number of Iterations')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('X-coordinate vs. Y-coordinate (Color represents Number of Iterations)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 2: Distance vs. Number of Iterations with a logarithmic fit
plt.figure(figsize=(10, 6))
plt.scatter(distance, its, alpha=0.75, edgecolor='k', s=100)
plt.xlabel('Distance to Minimum')
plt.ylabel('Number of Iterations')
plt.title('Distance to Minimum vs. Number of Iterations')

# Transform distance for fitting
transformer = FunctionTransformer(np.log, validate=True)
x_trans = transformer.fit_transform(np.array(distance).reshape(-1, 1))

# Fit the Linear Regression
regressor = LinearRegression()
results = regressor.fit(x_trans, its)
y_fit = results.predict(x_trans)

# Plot the fitted logarithmic function
plt.plot(distance, y_fit, color='red', linewidth=2, label='Logarithmic Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 3: Function Evaluation Difference vs. Number of Iterations with a logarithmic fit
plt.figure(figsize=(10, 6))
plt.scatter(function_evals, its, alpha=0.75, edgecolor='k', s=100)
plt.xlabel('Function Evaluation Difference')
plt.ylabel('Number of Iterations')
plt.title('Function Evaluation Difference vs. Number of Iterations')

# Transform function_evals for fitting
x_trans = transformer.fit_transform(np.array(function_evals).reshape(-1, 1))

# Fit the Linear Regression
results = regressor.fit(x_trans, its)
y_fit = results.predict(x_trans)

# Plot the fitted logarithmic function
plt.plot(function_evals, y_fit, color='red', linewidth=2, label='Logarithmic Fit')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Calculate the minimum of (x - minimum_x) and (y - minimum_y)
min_distances = np.minimum(np.abs(points[:, 0] - f1_min[0]), np.abs(points[:, 1] - f1_min[1]))

# Calculate the minimum of (x - minimum_x) and (y - minimum_y)
min_distances = np.minimum(np.abs(points[:, 0] - f1_min[0]), np.abs(points[:, 1] - f1_min[1]))

# Plot: Distance to Minimum vs. Minimum of (x - minimum_x, y - minimum_y)
plt.figure(figsize=(10, 6))
plt.scatter(min_distances, its, alpha=0.75, edgecolor='k', s=100)
plt.xlabel('Minimum of |x - minimum_x| and |y - minimum_y|')
plt.ylabel('Number of iterations')
plt.title('Minimum of |x - minimum_x| and |y - minimum_y| x Number of iterations')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()