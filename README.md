# faster monte-carlo
 use numba and parallel to accelerate mc simulation

The main purpose is to price an option by monte carlo method.

I use numba, vectorization and parallel to accelerate the loop, which is 100 times faster than simple loop.

The method still has some problems:

- vectorization does not accelerate the process compared with simple loop
- the use of numba is very limited
- muti-level parallel does not work

