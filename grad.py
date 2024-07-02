from typing import List, Callable

import numpy as np

from parameters import Parameters


def gradient(f: Callable[[List[float]], float], x: List[float], h=1e-6) -> List[float]:
    """Compute the gradient of a function f at point x using finite differences."""
    grad = [0.0] * len(x)
    fx = f(x)
    for i in range(len(x)):
        x_h = x[:]  # Create a copy of x
        x_h[i] += h  # Perturb only the i-th element
        grad[i] = (f(x_h) - fx) / h
    return grad


def gradient_descent_fixed(
    f: Callable[[List[float]], float],
    x: List[float],
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha: float = 1e-5
) -> int:
    """
    Performs gradient descent optimization to minimize `f` starting from `x`.

    Args:
        f (Callable[[List[float]], float]):
            The function to minimize. It takes a list of floats as input and returns a float.
        x (List[float]):
            The starting point for the gradient descent.
        max_iter (int, optional):
            The maximum number of iterations for the gradient descent. Default is 1000.
        tol (float, optional):
            The tolerance for convergence. If the norm of the gradient is less than this value, the function will return.
            Default is 1e-6.
        alpha (float, optional):
            The learning rate or step size for the gradient descent updates. Default is 1e-5.

    Returns:
        int:
            The number of iterations performed.
    """
    for iteration in range(max_iter):
        grad = gradient(f, x)
        if np.linalg.norm(grad) < tol:
            return iteration
        x = [x_i - alpha * grad_i for x_i, grad_i in zip(x, grad)]
    return max_iter


def spi(
        r: float,
        delta: float,
        f: Callable[[float], float],
        x_min: List[float],
        tol: float,
        tol_den: float,
        max_iter: int,
        replace_worst: bool = True
) -> int:
    """Perform SPI (Successive Parabolic Interpolation) to find the minimum of a function f."""
    t = r + delta
    s = r - delta

    f_r = f(r)
    f_s = f(s)
    f_t = f(t)

    for i in range(max_iter):

        if replace_worst:
            val = min(abs(f_s - f_t), abs(f_s - f_r), abs(f_r - f_t))
        else:
            val = f_s - f_t

        if i > 2 and abs(val) <= tol:
            x_min[0] = (s + t) / 2
            return i

        numerator = (f_s - f_r) * (t - r) * (t - s)
        denominator = 2 * ((s - r) * (f_t - f_s) - (f_s - f_r) * (t - s))

        if abs(denominator) < tol_den:
            x_val = (s + t + r) / 3
        else:
            x_val = ((r + s) / 2) - numerator / denominator

        f_x_val = f(x_val)

        if replace_worst:
            # Replace the worst of the three points
            if f_r >= f_s and f_r >= f_t:
                r, f_r = x_val, f_x_val
            elif f_s >= f_r and f_s >= f_t:
                s, f_s = x_val, f_x_val
            else:
                t, f_t = x_val, f_x_val
        else:
            # Replace the oldest of the three points
            r, f_r, s, f_s, t, f_t = s, f_s, t, f_t, x_val, f_x_val

    return 0


def g(alpha, f, x, v):
    return f([xi - alpha * vi for xi, vi in zip(x, v)])


def gradient_descent_spi(
        f: Callable[[List[float]], float],
        x: List[float],
        params: Parameters
) -> int:
    """
    Perform gradient descent optimization with variable step using SPI.

    Args:
        f (Callable[[float]], float): The function to minimize.
        x (List[float]): The initial point.
        params (Parameters): The parameters for gradient descent with SPI.

    Returns:
        int: The number of iterations performed.
    """
    for k in range(params.max_iter):
        v = gradient(f, x)

        x_min = [0]

        def g_alpha(alpha):
            return g(alpha, f, x, v)

        # Call SPI to find the optimal alpha
        spi(params.r, params.delta, g_alpha, x_min, params.mips_tol, params.mips_tol_den, params.max_iter,
            params.replace_worst)
        alfa = x_min[0]
        for i in range(len(x)):
            x[i] = x[i] - v[i] * alfa
        if (sum([i ** 2 for i in v])) ** (1 / 2) < params.tol:
            return k

    return params.max_iter
