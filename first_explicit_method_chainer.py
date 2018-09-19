# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:48:16 2018

@author: user
"""

import numpy as np

import chainer.functions as F
from chainer import Variable

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Define functions
def f(x): # Potential energy function
    return (x[0] + x[1])**4 + (x[0]/2 - x[1]/2)**4

def k(p, exponent=2): # Kinetic energy function
    return F.sum(F.absolute(p)**exponent)/exponent

# Initialization
i = 0
x_i = np.array([2., 1.])
p_i = np.array([0., 0.])

# Hyper parameters
gamma = 0.2
epsilon = 0.01
delta = 1/(1 + gamma*epsilon)

# Define arrays to save
f_store = np.array(f(x_i))
X = np.reshape(x_i, (1, -1))

print("Starting Optimization")
while(1): # Untill x converged
    """ First Explicit Method """
    # Compute grad f
    x_ivar = Variable(x_i)
    potential_energy = f(x_ivar)
    potential_energy.backward()
    grad_f = x_ivar.grad

    # Equation 1
    p_ip1 = delta*p_i - epsilon*delta*grad_f

    # Compute grad k
    p_ip1var = Variable(p_ip1)
    exponent = 4/3
    kinetic_energy = k(p_ip1var, exponent)
    kinetic_energy.backward()
    grad_k = p_ip1var.grad

    # Equation 2
    x_ip1 = x_i + epsilon*grad_k

    # Update i
    i += 1

    # Check convergence
    if np.all(abs(x_i - x_ip1) < 1e-6):
        break

    # Update x and p
    p_i = p_ip1
    x_i = x_ip1

    # print
    if i%1000==0:
        print("iter:", i,", f(x):", f(x_i))

    # Save results
    f_store = np.append(f_store, f(x_i))
    X = np.append(X, np.reshape(x_i, (1, -1)), axis=0)

print("Plotting results")
xmin, xmax, xstep = -2.5, 2.5, .2
ymin, ymax, ystep = -2.5, 2.5, .2

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
z = f([x, y])
minima = np.array([0, 0])
minima_ = minima.reshape(-1, 1)

fig, ax = plt.subplots(figsize=(10, 10))

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
#ax.quiver(X.T[0,:-1], X.T[1,:-1], X.T[0,1:]-X.T[0,:-1], X.T[1,1:]-X.T[1,:-1], scale_units='xy', angles='xy', scale=1, color='k')
ax.scatter(X.T[0], X.T[1], s=5)
ax.plot(*minima_, 'r*', markersize=5)

ax.set_xlabel('x')
ax.set_ylabel('y')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
I = np.arange(i)
ax.set_xlabel('x')
ax.set_ylabel('log[f(x)]')
plt.scatter(I, np.log(f_store), s=5)
plt.show()
