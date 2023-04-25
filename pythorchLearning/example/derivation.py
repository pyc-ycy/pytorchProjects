#! /anaconda3/python.exe
# author: pyc

import torch as th
from torch import autograd

x = th.tensor(1.)
a = th.tensor(1., requires_grad=True)
b = th.tensor(2., requires_grad=True)
c = th.tensor(3., requires_grad=True)

y = a**2 *x + b*x + c
print("before: ", a.grad, b.grad, c.grad)
grads = autograd.grad(y, [a, b, c])
print("after: ", grads[0], grads[1], grads[2])
