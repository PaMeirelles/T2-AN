from dataclasses import dataclass


@dataclass
class Parameters:
    """
    Dataclass for storing parameters for the gradient descent optimization.

    Attributes:
        r (int): An integer parameter, default is 0.
        max_iter (int): The maximum number of iterations, default is 1000.
        tol (float): The tolerance for convergence, default is 1e-6.
        delta (float): The initial step size, default is 1.0.
        mips_tol (float): The tolerance for the MIPS stopping criterion, default is 1e-6.
        mips_tol_den (float): The tolerance for the MIPS denominator, default is 1e-6.
    """
    r: int = 0
    max_iter: int = 1000
    tol: float = 1e-6
    delta: float = 1.0
    mips_tol: float = 1e-6
    mips_tol_den: float = 1e-6
