from typing import List, Callable

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
) -> List[float]:
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
        alpha (float, optional):a
            The learning rate or step size for the gradient descent updates. Default is 1e-5.

    Returns:
        List[float]:
            The point which is a local minimum of the function `f`.
    """
    for _ in range(max_iter):
        grad = gradient(f, x)
        if all(abs(gr) < tol for gr in grad):
            break
        for i in range(len(x)):
            x[i] -= alpha * grad[i]
    return x


def spi(
        r: float,
        delta: float,
        f: Callable[[float], float],
        x_min: List[float],
        tol: float,
        tol_den: float,
        max_iter: int
) -> int:
    """Perform SPI (Successive Parabolic Interpolation) to find the minimum of a function f."""
    t = r + delta
    s = r - delta

    f_r = f(r)
    f_s = f(s)
    f_t = f(t)

    for i in range(max_iter):
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

        r = s
        s = t

        t = x_val

        f_r = f_s
        f_s = f_t
        f_t = f(x_val)

    return 0


def g(alpha, f, x, v):
    return f([xi - alpha * vi for xi, vi in zip(x, v)])


def gradient_descent_spi(
        f: Callable[[List[float]], float],
        x: List[float],
        params: Parameters
) -> List[float]:
    """
    Perform gradient descent optimization with variable step using SPI

    Args:
        f (Callable[[float], float]): The function to minimize.
        x (float): The initial point.
        params (Parameters): The parameters for gradient descent with SPI.

    Returns:
        float: The optimized value of x.
    """
    for k in range(params.max_iter):
        v = gradient(f, x)

        x_min = [0]

        def g_alpha(alpha):
            return g(alpha, f, x, v)

        # Call MIPS to find the optimal alpha
        spi(params.r, params.delta, g_alpha, x_min, params.mips_tol, params.mips_tol_den, params.max_iter)
        alfa = x_min[0]
        for i in range(len(x)):
            x[i] = x[i] - v[i] * alfa
        if (sum([i ** 2 for i in v])) ** (1 / 2) < params.tol:
            return x

    return x
