# Getting Started with Python

## Installing Python

There are multiple ways to get a working python installation on your computer, depending on whether you're using Windows, Mac, or Linux.

The first option is to install standard python, which you can download [here](https://www.python.org/downloads/).

The second option is to install [Anaconda](https://docs.anaconda.com/free/anaconda/install/), which comes with a bunch of extra packages and tools that can be useful for scientific computing.

Personally, I don't use Anaconda as I prefer to have more control over what I install, but both options are popular.

### Resources

- [Installing python from the windows store](https://learn.microsoft.com/en-us/windows/python/beginners#install-python)

## Installing an IDE

An IDE is basically a text editor combined with some additional tools that allow you to run and debug your code.
My IDE of choice is [VSCode](https://code.visualstudio.com/).
Two other popular IDEs are [PyCharm](https://www.jetbrains.com/pycharm/) and [Spyder](https://www.spyder-ide.org/), both of these are more specifically designed for python programming.

```{tip}
Whatever IDE you choose, make sure you learn how to use it's python debugger.
A debugger will allow you to pause the program at a specific point and then step through the code line by line and inspect the values of variables.
Debugging this way is much more efficient than using print statements.
```

```{tip}
You should also learn to use some of the keyboard shortcuts for your IDE.
Different IDEs have different shortcuts that will make your editing experience easier by allowing you to quickly do things like rename a variable everywhere it appears in your code, comment and uncomment large blocks of code, and create multiple cursors to edit multiple lines at once.

VSCode has some walkthroughs of these features that should show up the first time you open it.
```

### Resources

- [Getting Started with Python in VS Code](https://code.visualstudio.com/docs/python/python-tutorial)
- [Very extensive (but a little old) video on using VSCode for python](https://www.youtube.com/watch?v=-nh9rCzPJ20)
- [Spyder video tutorial series](https://www.youtube.com/playlist?list=PLPonohdiDqg9epClEcXoAPUiK0pN5eRoc)
- [Getting started with PyCharm video series](https://www.youtube.com/playlist?list=PLCTHcU1KoD98IeuVcqJ2rt1FNytfR_C90)

## Installing Packages

Most python packages are installed through **pip** the default python package manager.
You can install packages by running `pip install <package name>` in the terminal.
If you installed python using anaconda/conda, you can also install packages using `conda install <package name>`, but not everything you can install with pip is available through conda.
You can also `pip install` things if you use conda, but make sure you have run `conda install pip` in the relevant environment first.

### tl;dr

To install the primary packages you'll need for this course, run the following in the terminal:

```shell
pip install --upgrade pip
pip install numpy scipy jaxlib jax[cpu] matplotlib niceplots
```

### [NumPy](https://numpy.org/doc/stable/)

NumPy is **THE**  standard for dealing with vectors, matrices, and multidimensional arrays in python.

### [JAX](https://jax.readthedocs.io/en/latest/index.html)

JAX is a library that does many things useful for machine learning.
Most importantly for this course, it has a module that allows you to compute derivatives through NumPy-like code.
[here](https://github.com/google/jax#pip-installation-cpu)

### [SciPy](https://docs.scipy.org/doc/scipy/)

SciPy contains a bunch of more complex scientific computing algorithms for things like root finding, optimization, solving ODE's, and sparse linear algebra.

### [Matplotlib](https://matplotlib.org/)

The standard plotting library for python.
It has a lot of functionality and excellent documentation.

```{tip}
If you're a Matlab user, you may be used to plotting simply by calling the `plot` function, and switching between different figures using the `figure` function.
It is possible to use a similar approach in Matplotlib, but I'd recommend that you instead use the approach used in the [matplotlib examples](https://matplotlib.org/stable/users/getting_started/) where you first create figure and axis objects and then call the plot methods of the axis.
This makes it much clearer in your code which data is being plotted on which figure.
```

### NicePlots

NicePlots is a little package I helped write with my labmates that we use to make our Matplotlib plots look nicer.
It is not at all necessary for the course, but if you do want to use it you can pip install it just like all the other packages mentioned already.
You can also check out some example uses of it [here](https://mdolab-niceplots.readthedocs-hosted.com/en/latest/auto_examples/index.html).

## Learning python

- [NumPy fundamentals](https://numpy.org/doc/stable/user/basics.html)
- [NumPy guide for Matlab users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)
- [Example code working with NumPy arrays](https://numpy.org/numpy-tutorials/content/tutorial-static_equilibrium.html)
