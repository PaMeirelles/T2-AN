import numpy as np
import grad
def f_1(x):
    return (x[0] ** 4) + (x[1] ** 4) - (2 * x[0] ** 2 * x[1] ** 2) + (6 * x[0]*x[1]) - (4 * x[0]) - (4 * x[1]) + 1
def f_r_2(x):
    return 100*((x[1]-x[0] ** 2)**2) + (x[0] -1)**2 

#testando o gradiente
x = [2,2]
print("teste\nteste\n")
print(f_r_2([2,2]))
print(grad.gradient_descent_fixed(f_r_2, x))
print(f_r_2(grad.gradient_descent_fixed(f_r_2, x)))

print("teste\nteste\n")
print(f_r_2([2,2]))
print(grad.gradient_descent_mips(f_1, x))
print(f_1(grad.gradient_descent_mips(f_1, x)))
