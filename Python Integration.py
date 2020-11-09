
import numpy as np
import sympy
import itertools
import math

def numint_py(f, a, b, n):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    "lb" scheme. This function must use pure Python, no Numpy.

    >>> round(numint_py(math.sin, 0, 1, 100), 5)
    0.45549
    >>> round(numint_py(lambda x: 1, 0, 1, 100), 5)
    1.0
    >>> round(numint_py(math.exp, 1, 2, 100), 5)
    4.64746

    """
    A = 0
    w = (b - a) / n # width of one slice
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    # iterate n times to calculate total area of region
    for i in range(n):
        # calculate x coordinate for each iteration 
        xi = a + i*w
        # calculate f(x) for the above x
        fxi = f(xi)
        # area of individual strip will be f(x)*w
        A += fxi*w
    # return total area of region
    return A

def numint(f, a, b, n, scheme='mp'):
    """Numerical integration. For a function f, calculate the definite
    integral of f from a to b by approximating with n "slices" and the
    given scheme. This function should use Numpy, and eg np.linspace()
    will be useful.
    
    >>> round(numint(np.sin, 0, 1, 100, 'lb'), 5)
    0.45549
    >>> round(numint(lambda x: np.ones_like(x), 0, 1, 100), 5)
    1.0
    >>> round(numint(np.exp, 1, 2, 100, 'lb'), 5)
    4.64746
    >>> round(numint(np.exp, 1, 2, 100, 'mp'), 5)
    4.67075
    >>> round(numint(np.exp, 1, 2, 100, 'ub'), 5)
    4.69417

    """
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    A = 0
    # divide the region into n parts 
    X = np.linspace(a,b,n+1)
    w = (b - a) / n # width of one slice
    
    # area for lower bound scheme
    if scheme == 'lb':
        Y = f(X)
        Y = Y[0:-1] # first to (n-1)th value 
        Y = Y*w
        A = np.sum(Y)
    # area for upper bound scheme
    elif scheme == 'ub': 
        Y = f(X)
        Y = Y[1:] # second to (n)th value
        Y = Y*w
        A = np.sum(Y)
    # area for mid point scheme
    else:
        for i in range(X.shape[0]-1):
            X[i]=(X[i]+X[i+1])/2 # storing consecutive averages 
        Y = f(X)
        Y = Y[:-1] # first to (n-1)th value 
        Y = Y*w
        A = np.sum(Y)
    # return total area of region
    return A

def true_integral(fstr, a, b):
    """Using Sympy, calculate the definite integral of f from a to b and
    return as a float. Here fstr is an expression in x, as a str. It
    should use eg "np.sin" for the sin function.

    This function is quite tricky, so you are not expected to
    understand it or change it! However, you should understand how to
    use it. See the doctest examples.

    >>> true_integral("np.sin(x)", 0, 2 * np.pi)
    0.0
    >>> true_integral("x**2", 0, 1)
    0.3333333333333333

    STUDENTS SHOULD NOT ALTER THIS FUNCTION.

    """
    x = sympy.symbols("x")
    # make fsym, a Sympy expression in x, now using eg "sympy.sin"
    fsym = eval(fstr.replace("np", "sympy")) 
    A = sympy.integrate(fsym, (x, a, b)) # definite integral
    A = float(A.evalf()) # convert to float
    return A

def numint_err(fstr, a, b, n, scheme):
    """For a given function fstr and bounds a, b, evaluate the error
    achieved by numerical integration on n points with the given
    scheme. Return the true value, absolute error, and relative error
    as a tuple.

    Notice that the relative error will be infinity when the true
    value is zero. None of the examples in our assignment will have a
    true value of zero.

    >>> print("%.4f %.4f %.4f" % numint_err("x**2", 0, 1, 10, 'lb'))
    0.3333 0.0483 0.1450

    """
    f = eval("lambda x: " + fstr) # f is a Python function
    A = true_integral(fstr, a, b)
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    # absolute difference between true integral and calculated numerical integral
    abs_error = abs(A - numint(f,a,b,n,scheme)) 
    # absolute error divided by the absolute value of the true value
    rel_error = abs_error/abs(A)
    
    return (A , abs_error , rel_error)

def make_table(f_ab_s, ns, schemes):
    """For each function f with associated bounds (a, b), and each value
    of n and each scheme, calculate the absolute and relative error of
    numerical integration and print out one line of a table. This
    function doesn't need to return anything, just print. Each
    function and bounds will be a tuple (f, a, b), so the argument
    f_ab_s is a list of tuples.

    Hint 1: use print() with the format string
    "%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g", or an equivalent f-string approach.
    
    Hint 2: consider itertools.

    >>> make_table([("x**2", 0, 1), ("np.sin(x)", 0, 1)], [10, 100], ['lb', 'mp'])
    x**2,0.00,1.00,10,lb,0.3333,0.04833,0.145
    x**2,0.00,1.00,10,mp,0.3333,0.0008333,0.0025
    x**2,0.00,1.00,100,lb,0.3333,0.004983,0.01495
    x**2,0.00,1.00,100,mp,0.3333,8.333e-06,2.5e-05
    np.sin(x),0.00,1.00,10,lb,0.4597,0.04246,0.09236
    np.sin(x),0.00,1.00,10,mp,0.4597,0.0001916,0.0004168
    np.sin(x),0.00,1.00,100,lb,0.4597,0.004211,0.009161
    np.sin(x),0.00,1.00,100,mp,0.4597,1.915e-06,4.167e-06

    """
   
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    for func,n,scheme in itertools.product(f_ab_s,ns,schemes):
        # using numint_err function to calculate true_value , absolute and relative error
        true_value,abs_error,rel_error = numint_err(func[0],func[1],func[2],n,scheme)
        print("%s,%.2f,%.2f,%d,%s,%.4g,%.4g,%.4g"%(func[0],func[1],func[2],n,scheme,true_value,abs_error,rel_error))

def main():
    """Call make_table() as specified in the pdf."""
    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    make_table([("np.cos(x)",0,np.pi),("np.sin(2*x)",0,1),("np.exp(x)",0,1)],[10,100,1000],['lb','mp'])

"""
    Result Interpretation :
        1. Observing the results, mid point scheme is better than lower bound scheme.
        2. For cos(x) absolute error when mid point scheme used equals 2.335e-16 and lower bound scheme used equals 0.003142 when n = 1000.
           Difference in absolute error between lb and mp scheme equals 0.0031419999999997665.
        3. Similiarly for sin(2x) and e^x the absolute error when mid point scheme used is lower than in lower bound scheme.
        4. Same pattern is observed in case of relative error ie. mp scheme has less error than lb scheme.
        5. Increasing n value decreses the absolute and relative errors, resulting in more accurate results.
        6. The variation in result is observed due to the fact that mp scheme creates a smoothing effect due to averaging of x coordinates.
"""

def numint_nd(f, a, b, n):
    """numint in any number of dimensions.

    f: a function of m arguments
    a: a tuple of m values indicating the lower bound per dimension
    b: a tuple of m values indicating the upper bound per dimension
    n: a tuple of m values indicating the number of steps per dimension

    STUDENTS ADD DOCTESTS
    
    >>> round(numint_nd(lambda x,y:x**2+y**2,(-2,-2),(2,2),(100,100)) ,5)
    42.6752
    
    >>> round(numint_nd(lambda x,y,z:x+y+z,(0,8,12),(7,10,13),(100,100,100)) ,5)
    349.3
    
    """

    # My implementation uses Numpy and the mid-point scheme, but you
    # are free to use pure Python and/or any other scheme if you prefer.
    
    # Hint: calculate w, the step-size, per dimension
    w = [(bi - ai) / ni for (ai, bi, ni) in zip(a, b, n)]

    # STUDENTS ADD CODE FROM HERE TO END OF FUNCTION
    A = np.array(a) # lower bounds of all dimensions
    B = np.array(b) # upper bounds of all dimensions
    N = np.array(n) # number of division of all dimensions
    W = (B-A)/N # increments of all dimension
    w_inc = np.prod(W) # represents increment in integral for an iteration
        
    mylist=[list(range(0,ni)) for ni in n]    
    Integral=0 # store the final integral value 
    
    # iterate all possible combinations and add individual integrals     
    for z in itertools.product(*mylist):
        n = np.array(z) 
        xi = A + n*W
        # f(*xi)*w_inc represents increase in integral by this combination ( similiar to 2D integrals )
        Integral = Integral + (f(*xi)*w_inc)  
    
    return Integral



if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()
    """
        References:
            1. https://math.libretexts.org/Courses/Mount_Royal_University/MATH_2200%3A_Calculus_for_Scientists_II/2%3A_Techniques_of_Integration/2.5%3A_Numerical_Integration_-_Midpoint%2C_Trapezoid%2C_Simpson's_rule
            2. Integrating function in two varaiables : https://www.youtube.com/watch?v=RIK3ZXVzEdM
    """
