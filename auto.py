# -*- coding: utf-8 -*-
"""
Created on Tue May 14 17:51:54 2019

@author: mirza
"""

import autograd.numpy as np
from autograd import grad
#from autograd import grad, multigrad

A = np.array([[3, 4, 5, 2],
                   [4, 4, 3, 3],
                   [5, 5, 4, 3]])
'''
def cost(W, H):
    pred = np.dot(W, H)
    C = np.sqrt(((pred - A).flatten() ** 2).mean(axis=None))
    return C

rank = 5
learning_rate=0.5
n_steps = 4000

grad_cost= multigrad(cost, argnums=[0,1])

m, n = A.shape

W =  np.abs(np.random.randn(m, rank))
H =  np.abs(np.random.randn(rank, n))

print "Iteration, Cost"
for i in range(n_steps):
    if i % 500 == 0:
        print "*"*20
        print i,",", cost(W, H)
    del_W, del_H = grad_cost(W, H)
    W =  W - del_W * learning_rate
    H =  H - del_H * learning_rate
    
    # Ensuring that W, H remain non-negative. This is also called projected gradient descent
    W[W<0] = 0
    H[H<0] = 0
'''    
def cost(X):
    return X**2

grad_cost = grad(cost)

print cost(4)
print grad_cost(4.0)
print grad_cost(5.0)
print grad_cost(6.0)
print grad_cost(10.0)
     