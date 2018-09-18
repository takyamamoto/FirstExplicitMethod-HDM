import numpy as np
import matplotlib.pylab as plt
import sympy

x = sympy.Symbol('x')
f = x**4 - 4*x**3 + 6*x**2 - 4*x
nabla_f = sympy.diff(f, x)

p = sympy.Symbol('x')
k = p**2
nabla_k = sympy.diff(k, p)

gamma = 0.2
epsilon = 0.01

delta = 1/ (1 + gamma * epsilon)

end = 8000

s = sympy.Symbol('s')
t = sympy.Symbol('t')

s = 0
t = 10

i = 0

F = np.array(f.subs(x, s))
X = np.array(s)

while i < end:
    t = delta*t - epsilon*delta*nabla_f.subs(x, s)
    s = s+ epsilon*nabla_k.subs(p, t)
    i = i+1
    F = np.append(F, f.subs(x, s))
    X = np.append(X, s)

print("estimated x that gives minimum of f is", s)
print("estimated minimal of f is", f.subs(x, s))

x = np.arange(end + 1)
y = np.array(F)
z = np.array(X)

plt.scatter(x, y, s=1)
plt.title(f)
plt.xlabel("Number of calculations")
plt.ylabel("Value of estimated minimum of f")
plt.show()

plt.scatter(x, z, s=1)
plt.title(f)
plt.xlabel("Number of calculations")
plt.ylabel("Value of estimated x that gives minimum of f")
plt.show()
