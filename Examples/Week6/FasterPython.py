#!/usr/bin/env python
# coding: utf-8

# # How to write fast (or slow) Python code

# In[16]:


import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import numpy as np


# # Avoid large python loops

# In[17]:


def matMult(A, B):
    C = np.zeros((A.shape[0], B.shape[1]))

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


N = 100
M = 200
K = 300
A = np.random.rand(N, M)
B = np.random.rand(M, K)

get_ipython().run_line_magic("timeit", "matMult(A,B)")


# Now let's compare that to NumPy's matrix-matrix multiplication with the same matrices:

# In[18]:


get_ipython().run_line_magic("timeit", "A @ B")


# Numpy is literally 10,000x faster than our python code!

# ## When using JAX, avoid modifying array values in place

# In[23]:


# Define the symbolic function q(x)
def q(x, y, Lx, Ly, kappa, coeff):
    return (x / Lx * (1 - x / Lx) + y / Ly * (1 - y / Ly)) * (2 * coeff)


def heat_conduction_2D_slow(Lx=2.0, Ly=1.0, Nx=10, Ny=5, kappa=0.5, coeff=1.0, T0=0.0):
    dx = Lx / Nx  # Grid spacing in the x-direction
    dy = Ly / Ny  # Grid spacing in the y-direction

    assert dx == dy, "dx must be equal to dy for the 9-point stencil to work"

    h = dx

    # Create a 2D grid of x and y coordinates
    x = jnp.linspace(0, Lx, Nx + 1)
    y = jnp.linspace(0, Ly, Ny + 1)
    Y, X = jnp.meshgrid(y, x)
    N = (Nx + 1) * (Ny + 1)
    rowOffset = Nx + 1

    # Create the Laplacian operator for 2D using finite differences
    A = jnp.zeros([N, N])
    b = jnp.zeros(N)

    for iy in range(1, Ny):
        for ix in range(1, Nx):
            row = iy * rowOffset + ix  # Current row in matrix
            x = ix * dx
            y = iy * dy

            A = A.at[row, row - rowOffset - 1].set(-1 / (4 * h**2))
            A = A.at[row, row - rowOffset].set(-1 / (2 * h**2))
            A = A.at[row, row - rowOffset + 1].set(-1 / (4 * h**2))
            A = A.at[row, row - 1].set(-1 / (2 * h**2))
            A = A.at[row, row].set(3 / h**2)
            A = A.at[row, row + 1].set(-1 / (2 * h**2))
            A = A.at[row, row + rowOffset - 1].set(-1 / (4 * h**2))
            A = A.at[row, row + rowOffset].set(-1 / (2 * h**2))
            A = A.at[row, row + rowOffset + 1].set(-1 / (4 * h**2))
            b = b.at[row].set(q(x, y, Lx, Ly, kappa, coeff) / kappa)

    # enforce boundary conditions
    for ix in range(Nx + 1):
        i = ix
        A = A.at[i, i].set(1.0)
        b = b.at[i].set(T0)
        i = Ny * (Nx + 1) + ix
        A = A.at[i, i].set(1.0)
        b = b.at[i].set(T0)
    for iy in range(Ny + 1):
        i = iy * (Nx + 1)
        A = A.at[i, i].set(1.0)
        b = b.at[i].set(T0)

        i = iy * (Nx + 1) + Nx
        A = A.at[i, i].set(1.0)
        b = b.at[i].set(T0)

    # Solve the linear system
    T = jnp.linalg.solve(A, b)
    T_out = jnp.reshape(T, (Nx + 1, Ny + 1), order="F")  # reshape into matrix
    return T_out, X, Y


get_ipython().run_line_magic("timeit", "heat_conduction_2D_slow(Nx=40, Ny=20)")


# In[24]:


def heat_conduction_2D_fast(Lx=2.0, Ly=1.0, Nx=40, Ny=20, kappa=0.5, coeff=1.0, T0=0.0):
    dx = Lx / Nx  # Grid spacing in the x-direction
    dy = Ly / Ny  # Grid spacing in the y-direction

    assert dx == dy, "dx must be equal to dy for the 9-point stencil to work"

    h = dx

    # Create a 2D grid of x and y coordinates
    x = jnp.linspace(0, Lx, Nx + 1)
    y = jnp.linspace(0, Ly, Ny + 1)
    Y, X = jnp.meshgrid(y, x)
    N = (Nx + 1) * (Ny + 1)
    rowOffset = Nx + 1

    # Initialise A matrix as identity so the boundary conditions are already set
    A = jnp.eye(N)

    # Initialise b vector to all T0 so that boundary condition rows are already set
    b = jnp.ones(N) * T0

    # Initialise lists that will store the row, column and value of each non-zero element in A related to the non-boundary nodes
    Arows = []
    Acols = []
    Avals = []
    bRows = []
    bVals = []

    weights = [
        -0.25 / h**2,
        -0.5 / h**2,
        -0.25 / h**2,
        -0.5 / h**2,
        3.0 / h**2,
        -0.5 / h**2,
        -0.25 / h**2,
        -0.5 / h**2,
        -0.25 / h**2,
    ]

    for iy in range(1, Ny):
        for ix in range(1, Nx):
            row = iy * rowOffset + ix  # Current row in matrix
            x = ix * dx
            y = iy * dy

            Arows += [row] * 9
            Acols += [
                row - rowOffset - 1,
                row - rowOffset,
                row - rowOffset + 1,
                row - 1,
                row,
                row + 1,
                row + rowOffset - 1,
                row + rowOffset,
                row + rowOffset + 1,
            ]
            Avals += weights
            bRows.append(row)
            bVals.append(q(x, y, Lx, Ly, kappa, coeff) / kappa)

    # Now actually set the values in the matrix all in one go
    A = A.at[jnp.array(Arows), jnp.array(Acols)].set(jnp.array(Avals))
    b = b.at[jnp.array(bRows)].set(jnp.array(bVals))

    # Solve the linear system
    T = jnp.linalg.solve(A, b)
    T_out = jnp.reshape(T, (Nx + 1, Ny + 1), order="F")  # reshape into matrix
    return T_out, X, Y


get_ipython().run_line_magic("timeit", "heat_conduction_2D_fast(Nx=40, Ny=20)")

T_slow = heat_conduction_2D_slow(Nx=40, Ny=20)[0]
T_fast = heat_conduction_2D_fast(Nx=40, Ny=20)[0]

results_match = jnp.allclose(T_slow, T_fast)
if results_match:
    print("The fast and slow versions of the code give the same results!")
else:
    print("The fast and slow versions of the code give different results!")
