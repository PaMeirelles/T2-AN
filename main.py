import grad
from parameters import Parameters

def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0]*x[1]) - (4 * x[0]) - (4 * x[1]) + 1


def f_r_2(x):
    return 100*((x[1]-x[0] ** 2)**2) + (x[0] - 1)**2


def f_r_3(x):
    return 100 * (x[1] - x[0]**2)**2 + (x[0] - 1)**2 + 100 * (x[2] - x[1]**2)**2 + (x[1] - 1)**2


inputs = [[0, 0], [2, 2], [3, 13], [1, 0], [20, -5]]
functions_2d = [("f1", f1), ("2D Rosenbrock function", f_r_2)]

print("Testing gradients for 2D functions")
for func_name, func in functions_2d:
    for inpt in inputs:
        print(f"Gradient for {func_name} in {inpt} = {[round(x, 4) for x in grad.gradient(func, inpt)]}")
    print()

inputs = [[0, 0, 0], [1, 2, 3], [1, -2, 4], [-5, -5, -5]]
functions_3d = [("3D Rosenbrock function", f_r_3)]

print("Testing gradients for 3D functions")
for func_name, func in functions_3d:
    for inpt in inputs:
        print(f"Gradient for {func_name} in {inpt} = {[round(x, 4) for x in grad.gradient(func, inpt)]}")
    print()


print(f"Fixed gradient f1 starting from origin: {grad.gradient_descent_fixed(f1, [0, 0])}")
print(f"SPI gradient descent f1 starting from origin: {grad.gradient_descent_spi(f1, [0, 0], Parameters())}")
print()

print(f"Fixed gradient 2D Rosenbrock function starting from origin: {grad.gradient_descent_fixed(f_r_2, [0, 0])}")
print(f"SPI gradient descent 2D Rosenbrock function starting from origin: "
      f"{grad.gradient_descent_spi(f_r_2, [0, 0], Parameters(delta=.1, max_iter=int(1e5)))}")
print()

print(f"Fixed gradient 3D Rosenbrock function starting from origin: {grad.gradient_descent_fixed(f_r_3, [0, 0, 0])}")
print(f"SPI gradient descent 3D Rosenbrock function starting from origin:"
      f" {grad.gradient_descent_spi(f_r_3, [0, 0, 0], Parameters(delta=.1, max_iter=int(1e5)))}")