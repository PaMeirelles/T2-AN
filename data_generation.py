import csv
import numpy as np
from tqdm import tqdm
import grad
from parameters import Parameters


def f3(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2 + 100 * (x[2] - x[1] ** 2) ** 2 + (x[1] - 1) ** 2


def f2(x):
    return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0] * x[1]) - (4 * x[0]) - (4 * x[1]) + 1


f1_min = [1.13263815658242, -0.46597244636103796]
f2_min = [1, 1]
f3_min = [1, 1, 1]


def generate(n_points, min_value, f, tol, delta, rng, replace_worst, method):
    dim = len(min_value)  # Determine the number of dimensions based on the minimum value

    def dist(p1, p2):
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    # Generate random points using Uniform distribution
    points = [np.random.uniform(min_value[i] - rng, min_value[i] + rng, n_points) for i in range(dim)]
    points = np.stack(points, axis=-1)

    # Compute distances, iterations, and function evaluation differences with a progress bar
    distance = [dist(point, min_value) for point in points]
    its = []
    function_evals = []

    for point in tqdm(points, desc="Computing iterations"):
        if method == "fixed":
            iterations = grad.gradient_descent_fixed(f, list(point.copy()), int(1e7), tol, delta)
        elif method == "spi":
            iterations = grad.gradient_descent_spi(f, list(point.copy()), Parameters(replace_worst=replace_worst, tol=tol, delta=delta, max_iter=int(1e6)))
        else:
            raise ValueError("Unknown method specified")
        its.append(iterations)
        function_evals.append(f(point) - f(min_value))

    # Sort distance, its, and function_evals based on distance
    sorted_indices = np.argsort(distance)
    distance = np.array(distance)[sorted_indices]
    its = np.array(its)[sorted_indices]
    function_evals = np.array(function_evals)[sorted_indices]
    points = np.array(points)[sorted_indices]

    # Generate filename
    function_name = f.__name__
    file_name = f"{function_name}_n{n_points}_tol{tol}_delta{delta}_rng{rng}_rw{replace_worst}_method{method}.csv"

    # Save data to CSV
    header = [f"x{i}" for i in range(dim)] + ["distance", "iterations", "function_evals"]
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for point, dist_val, iter_val, func_eval in zip(points, distance, its, function_evals):
            writer.writerow(list(point) + [dist_val, iter_val, func_eval])


# Example call
generate(1000, f3_min, f3, 1e-5, 1e-2, 1e-1, True, "spi")
