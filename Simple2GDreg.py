# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:10:33 2019

@author: mirza
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def lossfuncSqr(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += math.pow(X[i,j] - Xn[i,j], 2)
            e += 1
    return sum/e
    
def lossfuncAbs(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
            e += 1
    return sum/e

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

with open(path + 'h.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]

X1 = np.array(array2d)

with open(path + 'l.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]

X2 = np.array(array2d)

epoc = 100
alpha = 0.01
k = 10

m,n1 = X1.shape
m,n2 = X2.shape

print m, n1
print m, n2

W1 = np.random.rand(m,k)
H1 = np.random.rand(n1,k)

W2 = np.random.rand(m,k)
H2 = np.random.rand(n2,k)

W1 = W1/10.0
H1 = H1/10.0

W2 = W2/10.0
H2 = H2/10.0

print ('----------')

index = []
errSqrX1 = []
errAbsX1 = []
errSqrX2 = []
errAbsX2 = []

for e in range(epoc):
    alpha = 0.01/np.sqrt(e+1)

    W1t = np.transpose(W1)
    H1t = np.transpose(H1)
    
    W2t = np.transpose(W2)
    H2t = np.transpose(H2)
    
    W1n = W1 - alpha * (
            -2 * np.dot( (X1 - np.dot(W1, H1t)) , H1 ) + 2 * W1 
            )    
    H1n = H1 - alpha * (
            -2 * np.dot( np.transpose( X1 - np.dot(W1 , H1t) ), W1 ) + 2 * H1 
            )
    
    W2n = W2 - alpha * (
            -2 * np.dot( (X2 - np.dot(W2, H2t)) , H2 ) + 2 * W2 
            )
    H2n = H2 - alpha * (
            -2 * np.dot( np.transpose( X2 - np.dot(W2 , H2t) ), W2 ) + 2 * H2 
            )
    
    #print ('---------------------------------------------------------')
    
    errorSqrX1 = lossfuncSqr(X1, np.dot(W1, H1t))
    errorAbsX1 = lossfuncAbs(X1, np.dot(W1, H1t))
    errorSqrX2 = lossfuncSqr(X2, np.dot(W2, H2t))
    errorAbsX2 = lossfuncAbs(X2, np.dot(W2, H2t))
    
    index.append(e)
    
    errSqrX1.append(errorSqrX1)
    errAbsX1.append(errorAbsX1)
    errSqrX2.append(errorSqrX2)
    errAbsX2.append(errorAbsX2)
    
    W1 = W1n
    H1 = H1n
    
    W2 = W2n
    H2 = H2n
        
    if (e % 10 == 0):
        print e

print ('X1: ')
print np.dot(W1,np.transpose(H1))

print ('X2: ')
print np.dot(W2,np.transpose(H2))

plt.figure(1)
plt.plot(index,errSqrX1)
plt.title('Square Error X1')
plt.xlabel('Iteration')

plt.figure(2)
plt.plot(index,errAbsX1)
plt.title('Absolute Error X1')
plt.xlabel('Iteration')

plt.figure(3)
plt.plot(index,errSqrX2)
plt.title('Square Error X2')
plt.xlabel('Iteration')

plt.figure(4)
plt.plot(index,errAbsX2)
plt.title('Absolute Error X2')
plt.xlabel('Iteration')

plt.show

np.savetxt(path + "W1.csv", W1, delimiter=",")
np.savetxt(path + "W2.csv", W2, delimiter=",")
