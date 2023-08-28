#!/usr/bin/env python
# coding: utf-8

# # Testing jupyter notebooks

# ## Import the necessary packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import niceplots

plt.style.use(niceplots.get_style())


# ## Make a really cool plot
#
# Let's plot $\sin(2\pi x)$ and $\cos(2\pi x)$

# In[3]:


fig, ax = plt.subplots()
x = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * x)
y2 = np.cos(2 * np.pi * x)
ax.plot(x, y, label="$\sin(2\pi x)$", clip_on=False)
ax.plot(x, y2, label="$\cos(2\pi x)$", clip_on=False)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", r"$\frac{1}{4}$", r"$\frac{1}{2}$", r"$\frac{3}{4}$", "1"])
ax.set_yticks([-1, 0, 1])
niceplots.label_line_ends(ax)
niceplots.adjust_spines(ax)
