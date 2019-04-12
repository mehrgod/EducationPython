# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 18:21:31 2019

@author: mirza
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def lossfuncSqr(X,Wn,Hn):
    Xn = np.dot(Wn,Hn)
    m,n = X.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += math.pow(X[i,j] - Xn[i,j], 2)

    return sum/20
    
def lossfuncAbs(X,Wn,Hn):
    Xn = np.dot(Wn,Hn)
    m,n = X.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
    return sum/20

path = 'C:/Project/EDU/files/2013/example/Topic/60/'

with open(path + 'X1test.txt') as file:
    array1 = [[float(digit) for digit in line.split('\t')] for line in file]

X1 = np.array(array1)
    
epoc = 100
a = 0.01
k = 2

m,n = X1.shape

print X1.shape

W1 = np.random.rand(m,k)
H1 = np.random.rand(n,k)

#W = W/100.0
#H = H/100.0

print W1
print H1

print ('----------')

W1n = np.zeros((m,k))
H1n = np.zeros((n,k))

index = []
errSqr = []
errAbs = []

for e in range(epoc):
    a = 0.1/np.sqrt(e+1)
    
    W1n = W1 - a * ( -2 * np.dot( (X1 - np.dot(W1, np.transpose(H1))) , H1 ) + 2 * W1 )
    H1n = H1 - a * ( -2 * np.dot( np.transpose( X - np.dot(W1 , np.transpose(H)) ), W1 ) + 2 * H1 )
    
    errorSqr = lossfuncSqr(X,Wn,np.transpose(Hn))
    errorAbs = lossfuncAbs(X,Wn,np.transpose(Hn))
    index.append(e)
    errSqr.append(errorSqr)
    errAbs.append(errorAbs)
    
    W = Wn
    H = Hn
    
    if (e % 10 == 0):
        print e

print np.dot(Wn,np.transpose(Hn))


plt.figure(1)
plt.plot(index,errSqr)
plt.title('Square Error')
plt.xlabel('Iteration')
plt.figure(2)
plt.plot(index,errAbs)
plt.title('Absolute Error')
plt.xlabel('Iteration')


plt.show