import grad
from parameters import Parameters


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0] * x[1]) - (4 * x[0]) - (4 * x[1]) + 1


def f_r_2(x):
    return 100 * ((x[1] - x[0] ** 2) ** 2) + (x[0] - 1) ** 2


def f_r_3(x):
    return 100 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2 + 100 * (x[2] - x[1] ** 2) ** 2 + (x[1] - 1) ** 2


inputs_2d = [[0, 0], [2, 2], [3, 13], [1, 0], [20, -5]]
functions_2d = [("f1", f1), ("2D Rosenbrock function", f_r_2)]

print("Testing gradients for 2D functions")
for func_name, func in functions_2d:
    for inpt in inputs_2d:
        print(f"Gradient for {func_name} in {inpt} = {[round(x, 4) for x in grad.gradient(func, inpt)]}")
    print()

inputs_3d = [[0, 0, 0], [1, 2, 3], [1, -2, 4], [-5, -5, -5]]
functions_3d = [("3D Rosenbrock function", f_r_3)]

print("Testing gradients for 3D functions")
for func_name, func in functions_3d:
    for inpt in inputs_3d:
        print(f"Gradient for {func_name} in {inpt} = {[round(x, 4) for x in grad.gradient(func, inpt)]}")
    print()

# Testing gradient descent for 2D functions
x_2d_start = [0, 0]
iterations = grad.gradient_descent_fixed(f1, x_2d_start)
print(
    f"Fixed gradient descent f1 starting from [0, 0]: {x_2d_start}, iterations: {iterations}, min value: {f1(x_2d_start)}")
params_2d_spi = Parameters(delta=.1)
iterations = grad.gradient_descent_spi(f1, x_2d_start, params_2d_spi)
print(
    f"SPI gradient descent f1 starting from [0, 0]: {x_2d_start}, iterations: {iterations}, min value: {f1(x_2d_start)}")
print()

x_2d_start = [0, 0]
iterations = grad.gradient_descent_fixed(f_r_2, x_2d_start)
print(f"Fixed gradient descent 2D Rosenbrock function starting from [0, 0]: {x_2d_start}, iterations: {iterations}, "
      f"min value: {f_r_2(x_2d_start)}")
params_2d_spi = Parameters(delta=.01, max_iter=int(1e5))
iterations = grad.gradient_descent_spi(f_r_2, x_2d_start, params_2d_spi)
print(f"SPI gradient descent 2D Rosenbrock function starting from [0, 0]: {x_2d_start}, iterations: {iterations}, min "
      f"value: {f_r_2(x_2d_start)}")
print()

# Testing gradient descent for 3D functions
x_3d_start = [0, 0, 0]
iterations = grad.gradient_descent_fixed(f_r_3, x_3d_start)
print(
    f"Fixed gradient descent 3D Rosenbrock function starting from [0, 0, 0]: {x_3d_start}, iterations: {iterations}, min value: {f_r_3(x_3d_start)}")
params_3d_spi = Parameters(delta=.01, max_iter=int(1e5))
iterations = grad.gradient_descent_spi(f_r_3, x_3d_start, params_3d_spi)
print(
    f"SPI gradient descent 3D Rosenbrock function starting from [0, 0, 0]: {x_3d_start}, iterations: {iterations}, min value: {f_r_3(x_3d_start)}")
