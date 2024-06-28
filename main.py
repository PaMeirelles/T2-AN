import grad


def f1(x):
    return (x[0] ** 4) + (x[1] ** 4) + (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0]*x[1]) - (4 * x[0]) - (4 * x[1]) + 1


def f_r_2(x):
    return 100*((x[1]-x[0] ** 2)**2) + (x[0] -1)**2


inputs = [[0, 0], [2, 2], [0, 1], [1, 0], [20, -5]]

print("Testando gradientes")
for inpt in inputs:
    print(f"Gradient for f1 in {inpt} = {[round(x, 4) for x in grad.gradient(f1, inpt)]}")