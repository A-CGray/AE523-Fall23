#!/usr/bin/env python
# coding: utf-8

# In python, it is easy to create a alternative version of a function where some of the input values are fixed.

# In[5]:


# This function takes 3 inputs
def func(x, x0, y0):
    return x0 + y0 * x


print(f"{func(1, 2, 3)=}")

# Now I want to create a version of func where x0 and y0 are fixed and x is the only input
xFixed = 2
yFixed = 3


# I can do this by writing a new function that calls func with the fixed values
def another_func(x):
    return func(x, xFixed, yFixed)


print(f"{another_func(1)=}")

# There's another way to do the same thing using something called a lambda function
yet_another_func = lambda x: func(x, xFixed, yFixed)

print(f"{yet_another_func(1)=}")


# In[ ]:
