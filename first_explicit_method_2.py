# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:48:16 2018

@author: user
"""
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import elementwise_grad

# Define functions
def f(x):
    return (x[0] + x[1])**4 + (x[0]/2 - x[1]/2)**4

def k(p):
    return (3/4)*(p[0]**(4/3) + p[1]**(4/3))

def grad_k(p):
    return np.real(np.sign(p)*np.abs([(p[0])**(1/3), (p[1])**(1/3)]))
    #return (p[0]**2 + p[1]**2)/2

# Obtain gradient functions
grad_f = elementwise_grad(f)
#grad_k = elementwise_grad(k)

# Initialization
i = 0
x_i = np.array([2., 1.])
p_i = np.array([0. + 0j, 0. + 0j])

# Hyper parameters
gamma = 0.2
epsilon = 0.01
delta = 1/(1 + gamma*epsilon)

# Define arrays to save
F = np.array(f(x_i))
X = np.reshape(x_i, (1, -1))

print("Starting Optimization")
while(1): # Untill x converged
    p_ip1 = delta*p_i - epsilon*delta*grad_f(x_i)
    x_ip1 = x_i + epsilon*grad_k(p_ip1+0j)
    i += 1

    if np.all(abs(x_i - x_ip1) < 1e-6):
        break

    # Update x and p
    p_i = p_ip1
    x_i = x_ip1
    # Save results
    F = np.append(F, f(x_i))
    X = np.append(X, np.reshape(x_i, (1, -1)), axis=0)

print("Plotting results")
xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2

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
plt.scatter(I, np.log(F), s=5)
plt.show()
