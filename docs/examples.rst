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

>>> x = derive1(2.0) # we want the first derivative
>>> result = f(x)
>>> result.first_derivative
12

where `x = derive1(2.0)` constructs the correct dual number for us.

Let's compute the second and third derivatives!

>>> from num_dual import derive3
>>> x = derive3(2.0)
>>> result = f(x)
>>> print(f"f(x)   = {result}\nf'(x)  = {result.first_derivative}\nf''(x) = {result.second_derivative}\nf'''(x) = {result.third_derivative}")
f(x)    = 8 + 12v1 + 12v2 + 6v3
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
>>> x, y = derive2(0.5, 1.0)
>>> x, y
(0.5 + 1ε1 + 0ε2 + 0ε1ε2, 1 + 0ε1 + 1ε2 + 0ε1ε2)

`derive2(0.5, 1.0)` returns a tuple of hyperdual numbers where the correct non-real parts are set to unity.
Calling the Rosenbrock function with hyperdual numbers returns a hyperdual number containing the first partial derivatives and
the second partial derivative.

>>> result = rosen([x, y])
>>> result
56.5 + -151ε1 + 150ε2 + -200ε1ε2

The field `second_derivative` now returns a tuple where the first value is the first partial derivative w.r.t `x` and the second value
is the first partial derivative w.r.t. `y`. We can now compare our values to the analytical solution:

>>> d2 = np.array(result.first_derivative)
array([-151.,  150.])
>>> assert all(d2 == rosen_der([0.5, 1.0]))


Compute partial derivatives of multiple arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider a function that takes three arguments as input, one of which is vector valued.
Such a function is e.g. the helmholtz energy, denoted as :math:`a`, a function important in statistical mechanics.
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
>>>     partial_density = n * NAV / V
>>>     return RGAS * t * np.sum(n * (np.log(partial_density * de_broglie**3) - 1))

The specifics of the equation are not important, but note that besides `t` and `v` being scalar values,
`n` is a vector valued argument. Also, you can easily use mathematical expressions from `numpy`.

Now, we can compute different partial derivatives. For example, we can compute the first derivative with respect to `t` (temperature).

>>> s = helmholtz_energy(derive1(t), v, n, mw).first_derivative # negative entropy

Or the partial derivative with respect to the values of `n`:

>>> mu = helmholtz_energy(t, v, derive1(n), mw).first_derivative
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