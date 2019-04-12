# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:07:24 2019

@author: mirza
"""

import numpy as np

X = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

X = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

A = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
])

print A.shape

B = np.array([
    [5, 3, 0, 1 ,2],
    [4, 0, 0, 1 ,3],
    [1, 1, 0, 5 ,1],
    [1, 0, 0, 4, 4],
])

#print B.shape

#print np.multiply(A,B)
#print np.dot(A,B)
#print A * B

wd = X[:,:2]
print wd

print X[:,2:]


hc = X[:2,:]
print hc