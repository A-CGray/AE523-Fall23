#!/usr/bin/env python
# coding: utf-8

# # Iterative linear solvers

# In[1]:


import numpy as np
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import niceplots

plt.style.use(niceplots.get_style())
niceColors = niceplots.get_colors_list()

# Force the jupyter notebook to use vector graphics
import matplotlib_inline

matplotlib_inline.backend_inline.set_matplotlib_formats("pdf", "svg")


# In[2]:


# Define the symbolic function q(x)
def q(x, L):
    return jnp.sin(jnp.pi * x / L)


# ## Problem definition
#
# In this example we'll be solving the same 1D heat transfer equation as [last week's example](../Week3/FiniteDifferenceScheme.ipynb)
#
# The 1D domain spans $0 \le x \le L$ and is split into $N$ intervals of length $\Delta x = L/N$, this gives $N+1$ nodes in the grid. The temperatures at the nodes are $T_0, T_1, \ldots, T_N$. Dirichlet boundary conditions are applied at $x=0$ and $x=L$, such that $T_0 = 1$ and $T_N=4$.
#
# ![The finite-difference grid](../../images/FDDomain.svg)

# Using the central difference approximation for the second derivative, we wrote the finite difference equation at each node as:
#
# $$ -\kappa\frac{T_{i-1} - 2T_i + T_{i+1}}{\Delta x^2} = q(x_i) \Rightarrow -T_{i-1} + 2T_i - T_{i+1} = \frac{1}{\kappa} \Delta x^2 q(x_i)$$
#
# From this equation we can derive the residual at each node as:
#
# $$ r_i = \kappa\left(-T_{i-1} + 2T_i - T_{i+1}\right) - \Delta x^2 q(x_i)$$
#
# The residual is the quantity that must be zero at the solution, so we can use it as a measure of the error in our solution.
# Note that this is a different kind of error than we have discussed previously, it is not the error relative to the true solution of the PDE, but the error relative to the solution of the discretized system of equations we have created with our finite difference approximation.

# In[5]:


def computeResidual(u, q, kappa, dx):
    """Compute the residual of the 1D heat equation

    _extended_summary_

    Parameters
    ----------
    u : jax.numpy.ndarray
        Current state vector
    q : jax.numpy.ndarray
        Source term vector
    kappa : float
        Thermal conductivity
    dx : float
        Grid spacing

    Returns
    -------
    jax.numpy.ndarray
        Residual vector
    """
    dx2 = dx**2
    r = jnp.zeros_like(u)
    for ii in range(1, len(x) - 1):
        r = r.at[ii].set(kappa * (u[ii - 1] - 2 * u[ii] + u[ii + 1]) - dx2 * q[ii])

    return r


def computeNorm(r):
    """Compute the "normalized" norm of a vector

    Parameters
    ----------
    r : jax.numpy.ndarray
        Vector to compute the norm of
    """
    return jnp.linalg.norm(r) / jnp.sqrt(len(r))


# Let's compute the residual for an initial guess at the solution, we'll generate the initial guess by linearly interpolating between the boundary conditions:

# In[6]:


# Define the parameters
L = 2.0  # Length of domain
kappa = 0.5  # Thermal conductivity
Nx = 10  # Number of intervals
T0 = 1.0  # Left boundary condition
TN = 4.0  # Right boundary condition

u = jnp.linspace(T0, TN, Nx + 1)  # Initial guess (linearly interpolating between the boundary conditions)

x = jnp.linspace(0, L, Nx + 1)  # Grid points
dx = x[1] - x[0]  # Grid spacing
qVec = q(x, L)  # Source term

r = computeResidual(u, qVec, kappa, dx)  # Compute the residual
print(f"Residual norm: {jnp.linalg.norm(r):.2e}")


# Surprisingly enough, it's not zero, so we need to solve the equations.
#
# The first method we will use is the Jacobi iteration. In a Jacobi iteration, we update each node in the grid by rearranging the finite difference equation to solve for $T_i$:
#
# $$T_{i,new} = $$

# In[ ]:


def jacobiStep(u, q, kappa, dx):
    """Perform one Jacobi step

    Parameters
    ----------
    u : jax.numpy.ndarray
        Current state vector
    q : jax.numpy.ndarray
        Source term vector
    kappa : float
        Thermal conductivity
    dx : float
        Grid spacing

    Returns
    -------
    jax.numpy.ndarray
        Updated state vector
    """
    dx2 = dx**2
    uNew = jnp.zeros_like(u)
    for ii in range(1, len(x) - 1):
        uNew = uNew.at[ii].set(0.5 * (u[ii - 1] + u[ii + 1] + dx2 * q[ii]) / kappa)

    return uNew
