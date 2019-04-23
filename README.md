# diffcp

`diffcp` is a Python package for computing the derivative of a convex cone program, with respect to its problem data. The derivative is implemented as an abstract linear map, with methods for its forward application and its adjoint. 

The implementation is based on the calculations in our paper [Differentiating through a cone program](http://web.stanford.edu/~boyd/papers/diff_cone_prog.html).

### Installation
`diffcp` is available on Pip.

```python
pip install diffcp
```

The requirements are:
* [NumPy](https://github.com/numpy/numpy)
* [SciPy](https://github.com/scipy/scipy)
* [SCS](https://github.com/bodono/scs-python)
* Python 3.x

If you require Python 2 support, please file a Github issue.

### Cone programs
`diffcp` differentiates through a primal-dual cone program pair. The primal problem must be expressed as 

```
minimize        c'x
subject to      Ax + s = b
                s in K
```
where  `x` and `s` are variables, `A`, `b` and `c` are the user-supplied problem data, and `K` is a user-defined convex cone. The corresponding dual problem is

```
minimize        b'y
subject to      A'y + c == 0
                y in K^*
```

with dual variable `y`.

### Usage

`diffcp` exposes the function

```python
solve_and_derivative(A, b, c, cone_dict, warm_start=None, **kwargs).
```

This function returns a primal-dual solution `x`, `y`, and `s`, along with
functions for evaluating the derivative and its adjoint (transpose). These
functions respectively compute right and left multiplication of the derivative
of the solution map at `A`, `b`, and `c` by a vector.

#### Arguments
The arguments `A`, `b`, and `c` correspond to the problem data of a cone program.
* `A` must be a [SciPy sparse CSC matrix](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html).
* `b` and `c` must be NumPy arrays.
* `cone_dict` is a dictionary that defines the convex cone `K`.
* `warm_start` is an optional tuple `(x, y, s)` at which to warm-start SCS.
* `**kwargs` are keyword arguments to forward to SCS (e.g., `verbose=True`).

These inputs must conform to the [SCS convention](https://github.com/bodono/scs-python) for problem data. The keys in `cone_dict` correspond to the cones, with
* `diffcp.ZERO` for the zero cone,
* `diffcp.POS` for the positive orthant,
* `diffcp.SOC` for a product of SOC cones,
* `diffcp.PSD` for a product of PSD cones, and
* `diffcp.EXP` for a product of exponential cones.

The values in `cone_dict` denote the sizes of each cone; the values of `diffcp.SOC`, `diffcp.PSD`, and `diffcp.EXP` should be lists. The order of the rows of `A` must match the ordering of the cones given above. For more details, consult the [SCS documentation](https://github.com/cvxgrp/scs/blob/master/README.md).

#### Return value
The function `solve_and_derivative` returns a tuple

```python
(x, y, s, derivative, adjoint_derivative)
```

* `x`, `y`, and `s` are a primal-dual solution.

* `derivative` is a function that applies the derivative at `(A, b, c)` to perturbations `dA`, `db`, `dc`. It has the signature 
```derivative(dA, db, dc) -> dx, dy, ds```, where `dA` is a SciPy sparse CSC matrix with the same sparsity pattern as `A`, and `db` and `dc` are NumPy arrays. `dx`, `dy`, and `ds` are NumPy arrays, approximating the change in the primal-dual solution due to the perturbation.

* `adjoint_derivative` is a function that applies the adjoint of the derivative to perturbations `dx`, `dy`, `ds`. It has the signature 
```adjoint_derivative(dx, dy, ds) -> dA, db, dc```, where `dx`, `dy`, and `ds` are NumPy arrays.

#### Example
```python
import numpy as np
from scipy import sparse

import diffcp

cone_dict = {
    diffcp.ZERO: 3,
    diffcp.POS: 3,
    diffcp.SOC: [5]
}

m = 3 + 3 + 5
n = 5

A, b, c = diffcp.utils.random_cone_prog(m, n, cone_dict)
x, y, s, D, DT = diffcp.solve_and_derivative(A, b, c, cone_dict)

# evaluate the derivative
nonzeros = A.nonzero()
data = 1e-4 * np.random.randn(A.size)
dA = sparse.csc_matrix((data, nonzeros), shape=A.shape)
db = 1e-4 * np.random.randn(m)
dc = 1e-4 * np.random.randn(n)
dx, dy, ds = D(dA, db, dc)

# evaluate the adjoint of the derivative
dx = c
dy = np.zeros(m)
ds = np.zeros(m)
dA, db, dc = DT(dx, dy, ds)
```

For more examples, including the SDP example described in the paper, see the [`examples`](examples/) directory.

### Citing
If you wish to cite `diffcp`, please use the following BibTex:

```
@article{diffcp2019,
    author       = {Agrawal, A. and Barratt, S. and Boyd, S. and Busseti, E. and Moursi, W.},
    title        = {Differentiation through a Cone Program},
    journal      = {arXiv preprint arXiv:1904.09043},
    year         = {2019},
}

@misc{diffcp,
    author       = {Agrawal, A. and Barratt, S. and Boyd, S. and Busseti, E. and Moursi, W.},
    title        = {{diffcp}: differentiation of a cone program, version 1.0},
    howpublished = {\url{https://github.com/cvxgrp/diffcp}},
    year         = 2019
}
```
