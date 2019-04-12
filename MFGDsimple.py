# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:04:59 2019

@author: mirza
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def lossfuncSqr(X,Wn,Wh):
    Xn = np.dot(Wn,Hn)
    m,n = X.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += math.pow(X[i,j] - Xn[i,j], 2)

    return sum/20
    
def lossfuncAbs(X,Wn,Wh):
    Xn = np.dot(Wn,Hn)
    m,n = X.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
    return sum/20

path = 'C:/Project/EDU/files/2013/example/Topic/60/'

with open(path + 'X1.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]

X = np.array(array2d)
'''
X = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
'''

epoc = 5
alpha = 0.01
k = 2

m,n = X.shape

print X.shape

W = np.random.rand(m,k)
H = np.random.rand(k,n)

Wn = np.zeros((m,k))
Hn = np.zeros((k,n))

index = []
errSqr = []
errAbs = []

for e in range(epoc):
    alpha = 0.1/np.sqrt(e+1)
    print alpha
    Wn = W + 2 * alpha * np.dot((X - np.dot(W,H)), np.transpose(H))
    Hn = H + 2 * alpha * np.dot(np.transpose(W), (X - np.dot(W,H)))
    
    print np.dot(Wn,Hn)
    #print ('---------------------------------------------------------')
    errorSqr = lossfuncSqr(X,Wn,Hn)
    errorAbs = lossfuncAbs(X,Wn,Hn)
    index.append(e)
    errSqr.append(errorSqr)
    errAbs.append(errorAbs)
    
    W = Wn
    H = Hn

print np.dot(Wn,Hn)


plt.figure(1)
plt.plot(index,errSqr)
plt.title('Square Error')
plt.xlabel('Iteration')
plt.figure(2)
plt.plot(index,errAbs)
plt.title('Absolute Error')
plt.xlabel('Iteration')


plt.show