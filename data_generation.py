import csv

import numpy as np
from tqdm import tqdm
import grad
from parameters import Parameters


def f_r_2(x):
    return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0] * x[1]) - (4 * x[0]) - (4 * x[1]) + 1


def generate(n_points, min_value, f, tol, delta, file_name, rng, replace_worst):
    x_range = [min_value[0] - rng, min_value[0] + rng]
    y_range = [min_value[1] - rng, min_value[1] + rng]
    np.random.seed(12)

    def dist(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # Generate random points using Uniform distribution
    points = [[np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])] for _ in
              range(n_points)]

    # Compute distances, iterations, and function evaluation differences with a progress bar
    distance = [dist(point, min_value) for point in points]
    its = []
    function_evals = []

    for point in tqdm(points, desc="Computing iterations"):
        iterations = grad.gradient_descent_spi(f, list(point.copy()),
                                               Parameters(replace_worst=replace_worst, tol=tol, delta=delta, max_iter=int(1e6)))
        its.append(iterations)
        function_evals.append(f(point) - f(min_value))

    # Sort distance, its, and function_evals based on distance
    sorted_indices = np.argsort(distance)
    distance = np.array(distance)[sorted_indices]
    its = np.array(its)[sorted_indices]
    function_evals = np.array(function_evals)[sorted_indices]
    points = np.array(points)[sorted_indices]

    # Save data to CSV
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "distance", "iterations", "function_evals"])
        for point, dist_val, iter_val, func_eval in zip(points, distance, its, function_evals):
            writer.writerow([point[0], point[1], dist_val, iter_val, func_eval])


f1_min = [1.13263815658242, -0.46597244636103796]
generate(1000, f1_min, f1, 5*1e-6, .1, "f1_replace_oldest_51e-6_tol_1e-1_delta_1e-2_range.csv", 1e-2, False)
# generate(1000, [1, 1], f_r_2, 1e-5, .01, "f2_1e-5tol.csv", 1e-4, True)
