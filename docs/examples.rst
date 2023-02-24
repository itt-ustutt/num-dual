Examples
--------

First, second and third derivatives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let's define a simple, scalar valued function for which we want to compute the first, second and third derivative.

>>> def f(x):
>>> """f'(x) = 3 * x**2, f''(x) = 6 * x"""
>>>     return x**3

The function is defined just like a regular python function.
Different from a regular python function though, we use (hyper)dual numbers as arguments.
For example, to compute the first derivative at x = 2, we need to call the function with a dual number as input, setting the dual part (ε)
to 1.0.

>>> from num_dual import Dual64
>>> x = Dual64(2.0, 1.0)
>>> x
2 + [1]ε

Then, calling the function, the result is also a dual number, where the real part (or value)
is the result of the function that we would get by simply calling it with a floating point number,
whereas the dual part (or first derivative) contains the derivative.

>>> result = f(x)
>>> result
8 + 12ε
>>> result.value
8
>>> result.first_derivative
12

The value we used for the dual part (1.0) is not important, however,
the resulting derivatives will be multiples of the chosen value and as such we set it to unity.

The procedure as outlined above works fine, but you have to know what type of dual number you have to use.
E.g. for the second derivative, the function argument has to be a hyerdual number. We therefore introduce
helper functions that can be used to simply declare the order of the derivative you want to compute.

The same result as above can be created via

>>> from num_dual import first_derivative
>>> (result, derivative) = first_derivative(f, 2.0)
>>> derivative
12

Internally, the correct dual number is constructed and used to evaluate the function.

Let's compute the second and third derivatives!

>>> from num_dual import third_derivative
>>> (f0, fx, fxx, fxxx) = third_derivative(f, 2.0)
>>> print(f"f(x)   = {f0}\nf'(X)  = {fx}\nf''(x) = {fxx}\nf'''(x)= {fxxx}")
f(x)    = 8.0
f'(x)   = 12.0
f''(x)  = 12.0
f'''(x) = 6.0

Partial derivatives
^^^^^^^^^^^^^^^^^^^

Hyperdual numbers can be used to compute partial derivatives of multivariate functions (functions of several real valued variables) as well.
A function that is often used as benchmark in optimization problems is the Rosenbrock function which is defined as:

  .. math::

    f(x,y) = (a - b)^2 + b(y - x^2)^2 \,,

The function and its derivatives are implemented in `scipy`. Let's compute the partial derivatives of the Rosenbrock function.

>>> from num_dual import derive2
>>> from scipy.optimize import rosen, rosen_der
>>> import numpy as np
>>> x, y = 0.5, 1.0

`seond_partial_derivative` can be used to calculate partial derivatives of bivariate functions (functions with two input parameters. Because the Rosenbrock function takes one single vector of inputs, it is wrapped in a lambda function.

>>> second_partial_derivative(lambda x, y: rosen([x, y]), x, y)
(56.5, -151.0, 150.0, -200.0)

The resulting tuple contains the function value, the first partial derivative w.r.t. `x`, the first partial derivative w.r.t. `y`, and the second partial derivative. We can now compare our values to the analytical solution:

>>> (f0, fx, fy, fxy) = second_partial_derivative(lambda x, y: rosen([x, y]), x, y)
>>> assert all(np.array([fx, fy]) == rosen_der([x, y]))

For multivariate functions, the gradient can also be computed directly

>>> from num_dual import gradient
>>> (_, g) = gradient(rosen, [0.5, 1.0])
>>> g
[-151.0, 150.0]


Compute partial derivatives of multiple arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a function that takes three arguments as input, one of which is vector valued.
Such a function is e.g. the helmholtz energy, denoted as :math:`A`, a function important in statistical mechanics.
It is a function of the volume, :math:`V`, the temperature, :math:`T`, and the number of particles of possibly
multiple components, :math:`N`. Let's define this function (for an ideal gas) and compute some interesting properties.

>>> from num_dual import *
>>> import numpy as np
>>>
>>> t = 300.0
>>> v = 20.0
>>> n = np.array([3, 2])
>>> mw = np.array([39.948e-3, 4e-3])
>>>
>>> def helmholtz_energy(t, v, n, mw):
>>>     H = 6.62607015e-34
>>>     NAV = 6.02214076e23
>>>     RGAS = 8.314
>>>     if isinstance(n, list):
>>>         n = np.array(n)
>>>     de_broglie = H * NAV / np.sqrt(2.0 * np.pi * mw * RGAS * t)
>>>     partial_density = n * NAV / v
>>>     return RGAS * t * np.sum(n * (np.log(partial_density * de_broglie**3) - 1))

The specifics of the equation are not important, but note that besides `t` and `v` being scalar values,
`n` is a vector valued argument. Also, you can easily use mathematical expressions from `numpy`.

Now, we can compute different partial derivatives. For example, we can compute the first derivative with respect to `t` (temperature).

>>> (_, s) = first_derivative(lambda t: -helmholtz_energy(t, v, n, mw), t) # entropy
>>> s
956.4722861925324

Or the partial derivative with respect to the values of `n`:

>>> (_, mu) = first_derivative(lambda n: helmholtz_energy(t, v, n, mw), n) # chemical potential
>>> mu
[-54192.23064420561, -46593.74696257142]


Compatibility with `numpy`
^^^^^^^^^^^^^^^^^^^^^^^^^^
The examples shown above contain very simple mathematical equations.
We provide evaluations for a lot of useful mathematical expressions that are defined in `numpy`.

>>> def f(x):
...     return np.exp(x) / np.sqrt(np.sin(x)**3 + np.cos(x)**3)
>>> f(1.5)
4.497780053946161

Calling the same function with a hyper dual number and dual parts of 1
yields the first and second derivatives. (ε1 and ε2 parts are identical)

>>> from num_dual import HyperDual64 as HD64
>>> x = HD64(1.5, 1.0, 1.0, 0.0)
>>> f(x)
4.497780053946162 + 4.053427893898621ε1 + 4.053427893898621ε2 + 9.463073681596605ε1ε2