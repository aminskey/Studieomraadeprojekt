import torch
import random
import matplotlib.pyplot as plt

from torch.distributions.multinomial import Multinomial

'''
# Vector
x = torch.arange(4.0)
#print(x)

# Matrix
A = torch.arange(6).reshape(2, 3)
#print(A, A.T.T, A==A.T.T)

# Exercise 2
B = torch.arange(6, 12).reshape(2, 3)
X = A.T + B.T

#print(f"A: {A}\nB: {B}\nX: {X}\n(A+B)T {(A+B).T}\n")
'''

# Automatic differentiation
x = torch.arange(4.0)
x.requires_grad_(True)

# x.grad is by default none
print(x.grad)

# assign y (function of x)
# y = 2 x.x or y = 2*(x T x)
y = 2 * torch.dot(x, x)
print(y)

# now we calculate the gradient of y with respect to x
# by calling the backward method.
y.backward()
print(x.grad) # Now we have the gradient of y stored in x.grad

# We already know that the gradient of y=2x.x with respect to x should be 4x.
# Let's verify.
print(f"x.grad == 4*x\t+{x.grad == 4*x}")

# resets the gradient, instead of adding a new gradient
# to the already stored gradient
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# backward for non-scalar variables
x.grad.zero_()
y = x ** 2
# this will crash
# y.backward()
# RuntimeError: grad can be implicitly created only for
# scalar outputs

y.backward(gradient=torch.ones(len(y)))
# Can also be written as
# y.sum().backward()
print(x.grad)

# Detaching computation
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(f"x.grad == u\t{x.grad == u}")
# Detach helps in memory management,
# since it doesn't track the gradient of u
# so u.backward() isn't a thing
# but this doesn't limit y
x.grad.zero_()
y.sum().backward()
print(f"x.grad == 2*x\t{x.grad == 2*x}")

# The thing about automatic differentiation, is
# that even if, the function had to go through a whole ton of conditions
# or auxiliary variables, it would still work
# Check it out
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b *= 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a, d)
"""
    Even though our function f is, 
    for demonstration purposes, 
    a bit contrived, 
    its dependence on the input is quite simple: 
    
    it is a linear function of a with piecewise defined scale. 
    As such, f(a) / a is a vector of constant entries and, 
    moreover, f(a) / a needs to match the gradient of f(a) 
    with respect to a.
"""
print(f"a.grad == d/a \t{a.grad == d/a}")

# Probability and statistics
# Probability = chance of something happening
# Statistics = collected data, to describe a broader population
# Coin toss example where prob=50%
n = 100
heads = sum([random.random() > 0.5 for i in range(n)])
tails = n - heads
print(f"\nCoin toss:\ntosses : {n}\theads, tails: {[heads, tails]}")

# In this case, PyTorch's multinomial can do all of that
# in 2 lines
fair_probs = torch.tensor([0.5, 0.5])
print(f"Using multinomial: {Multinomial(n, fair_probs).sample()}")

#Plotting coin toss
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts/cum_counts.sum(dim=1, keepdim=True)
estimates = estimates.numpy()

plt.plot(estimates[:, 0], label=("P(coin=heads)"))
plt.plot(estimates[:, 1], label=("P(coin=tails)"))
plt.axhline(y=0.5, color='black', linestyle='dashed')

plt.gca().set_xlabel('samples')
plt.gca().set_ylabel('Estimated probability')

plt.legend()
plt.grid()
plt.show()