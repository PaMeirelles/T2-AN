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


# Read data from CSV
points, distance, its, function_evals = read_csv('f1.csv')

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
