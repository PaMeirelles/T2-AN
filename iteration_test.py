import random
import matplotlib.pyplot as plt
from math import sqrt
import grad
from parameters import Parameters
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression


# Function definition for the 2D Rosenbrock function
def f_r_2(x):
    return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0] * x[1]) - (4 * x[0]) - (4 * x[1]) + 1


# Constants
NUM_POINTS = 250
MIN = [1, 1]
X_RANGE = [MIN[0] - 2, MIN[0] + 2]
Y_RANGE = [MIN[1] - 2, MIN[1] + 2]
f = f_r_2
# Set the random seed for reproducibility
random.seed(12)


# Distance function
def dist(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Generate random points
points = [[random.uniform(X_RANGE[0], X_RANGE[1]), random.uniform(Y_RANGE[0], Y_RANGE[1])] for _ in range(NUM_POINTS)]

# Compute distances, iterations, and function evaluation differences with a progress bar
distance = [dist(point, MIN) for point in points]
its = []
function_evals = []

for point in tqdm(points, desc="Computing iterations"):
    iterations = grad.gradient_descent_spi(f_r_2, point.copy(), Parameters(delta=0.01, max_iter=int(1e6)))
    its.append(iterations)
    function_evals.append(f_r_2(point) - f_r_2(MIN))

# Plot 1: x-coordinate vs. y-coordinate with number of iterations as color bar
x_coords = [point[0] for point in points]
y_coords = [point[1] for point in points]

plt.figure(figsize=(10, 6))
sc = plt.scatter(x_coords, y_coords, c=its, cmap='viridis', alpha=0.75, edgecolor='k', s=100)
plt.colorbar(sc, label='Number of Iterations')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('X-coordinate vs. Y-coordinate (Color represents Number of Iterations)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 2: Distance vs. Number of Iterations with a linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(distance, its, alpha=0.75, edgecolor='k', s=100)
plt.xlabel('Distance to Minimum')
plt.ylabel('Number of Iterations')
plt.title('Distance to Minimum vs. Number of Iterations')

# Linear regression
X = np.array(distance).reshape(-1, 1)
y = np.array(its)
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
plt.plot(distance, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Plot 3: Function Evaluation Difference vs. Number of Iterations with a log scale on x-axis
plt.figure(figsize=(10, 6))
plt.scatter(function_evals, its, alpha=0.75, edgecolor='k', s=100)
plt.xscale('log')
plt.xlabel('Function Evaluation Difference (log scale)')
plt.ylabel('Number of Iterations')
plt.title('Function Evaluation Difference vs. Number of Iterations')

# Linear regression
X = np.log(np.array(function_evals).reshape(-1, 1) + 1)  # Log scale and avoid log(0)
y = np.array(its)
reg = LinearRegression().fit(X, y)
y_pred = reg.predict(X)
plt.plot(function_evals, y_pred, color='red', linewidth=2, label='Linear Regression')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
