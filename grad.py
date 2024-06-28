
#deron_f/deron_xi = (f(x, xi + h) - f(x, xi - h)) / 2h
def numeric_partial_derivative(f, x0, i, h=1e-2):
    """Compute the i-th partial derivative of f at x."""
    x = x0.copy()
    x[i] += h
    a = f(x)
    x[i] -= 2*h
    b = f(x)
    x[i] += h
    return (a - b) / (2*h)

def gradient(f, x, h=1e-6):
    """Compute the gradient of f at x using finite differences."""
    grad = [0] * len(x)
    fx = f(x)
    for i in range(len(x)):
        x_h = x[:]
        x_h[i] += h
        grad[i] = (f(x_h) - fx) / h
    return grad

def gradient_descent_fixed(f, x, max_iter=1000, tol=1e-6, alfa = 0.00001):
    """Perform gradient descent optimization."""
    #v = x[:]
    for k in range(max_iter):
        #for j in range(len(x)):
        #    v[j] = numeric_partial_derivative(f, x, j)
        v = gradient(f, x)
        for i in range(len(x)):
            x[i] = x[i] - v[i]*alfa
        if (sum([i**2 for i in v]))**(1/2) < tol:
            return x
        
    return x
def g(alpha, f, x, v):
    return f([xi - alpha * vi for xi, vi in zip(x, v)])

def mips(r, delta, f, xmin, TOL, TOL_DEM, max_iter):
    t = r + delta
    s = r - delta
    i = 0
    f_r = f(r)
    f_s = f(s)
    f_t = f(t)
    while i < max_iter:
        val = f_s - f_t
        if i > 2 and abs(val) <= TOL:
            xmin[0] = (s + t) / 2
            return i
        numerador = (f_s - f_r) * (t - r) * (t - s)
        denominador = 2 * ((s - r) * (f_t - f_s) - (f_s - f_r) * (t - s))
        if abs(denominador) < TOL_DEM:
            x_val = (s + t + r) / 3
        else:
            x_val = ((r + s) / 2) - numerador / denominador
        r = s
        s = t
        t = x_val
        f_r = f_s
        f_s = f_t
        f_t = f(x_val)
        i += 1
    if i >= 50:
        i = 0
    return i

def gradient_descent_mips(f, x, max_iter=1000, tol=1e-6, delta=1.0, TOL=1e-6, TOL_DEM=1e-6):
    """Perform gradient descent optimization."""
    #v = x[:]
    for k in range(max_iter):
        #for j in range(len(x)):
        #    v[j] = numeric_partial_derivative(f, x, j)
        v = gradient(f, x)


        r = 0
        xmin = [0]
        
        # Define the function g(alpha) using a lambda to capture x and v
        g_alpha = lambda alpha: g(alpha, f, x, v)
        
        # Call MIPS to find the optimal alpha
        mips(r, delta, g_alpha, xmin, TOL, TOL_DEM, max_iter)
        alfa = xmin[0]
        for i in range(len(x)):
            x[i] = x[i] - v[i]*alfa
        if (sum([i**2 for i in v]))**(1/2) < tol:
            return x
        
    return x