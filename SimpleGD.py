# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:26:40 2019

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
    #E = np.dot(Wn,Wh) - X
    #print np.sum(E)
    #print sum
    return sum/20
    
def lossfuncAbs(X,Wn,Wh):
    Xn = np.dot(Wn,Hn)
    m,n = X.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
    return sum/20

X = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
    

#W = np.random.rand(5,2)
#H = np.random.rand(2,4)

epoc = 20
alpha = 0.01
k = 2

m,n = X.shape

#print m
#print n

W = np.random.rand(m,k)
H = np.random.rand(k,n)
#W = np.zeros((m,k))
#H = np.zeros((k,n))

Wn = np.zeros((m,k))
Hn = np.zeros((k,n))

#print np.transpose(X)[0,1]

#print (X[1,0])
#print X.shape
#print W.shape

#print Wn

#print np.dot(W,H)
#print X - np.dot(W,H)
#print np.dot((X - np.dot(W,H)), np.transpose(H))
#print np.multiply((np.dot((X - np.dot(W,H)), np.transpose(H))),2)
#print 2 * np.dot((X - np.dot(W,H)), np.transpose(H))
#print 2 * alpha * np.dot((X - np.dot(W,H)), np.transpose(H))
#print (2 * alpha * np.dot((X - np.dot(W,H)), np.transpose(H))).shape
index = []
errSqr = []
errAbs = []

for e in range(epoc):
    alpha = 0.1/np.sqrt(e+1)
    print alpha
    Wn = W + 2 * alpha * np.dot((X - np.dot(W,H)), np.transpose(H))
    Hn = H + 2 * alpha * np.dot(np.transpose(W), (X - np.dot(W,H)))
    
    #print Wn
    #print Hn
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

#plt.plot(index,errSqr)
#plt.plot(index,errAbs)

plt.figure(1)
#plt.subplot(1)
plt.plot(index,errSqr)
plt.title('Square Error')
plt.xlabel('Iteration')
#plt.subplot(2)
plt.figure(2)
plt.plot(index,errAbs)
plt.title('Absolute Error')
plt.xlabel('Iteration')


plt.show

'''
    
    for i in range(m):
        for j in range(k):
            print W[i,j]
            Wn[i,j] = W[i,j] - 2 * alpha * (X[i,j] - W[i,j] * H[i,j]) * np.transpose(H)[i,j]
    
    for i in range(k):
        for j in range(n):
            Hn[i,j] = H[i,j] - 2 * alpha * (X[i,j] - W[i,j] * H[i,j]) * np.transpose(W)[i,j]
            
            '''