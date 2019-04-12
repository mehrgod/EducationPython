# -*- coding: utf-8 -*-
"""
Created on Wed Apr 03 18:06:53 2019

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

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

with open(path + 'l.txt') as file:
    array1d = [[float(digit) for digit in line.split('\t')] for line in file]

with open(path + 'h.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]

X1 = np.array(array1d)
X2 = np.array(array2d)
   
epoc = 100
#alpha = 0.01
k = 10

m,n = X1.shape
print ('X1 shape')
print X1.shape

W1 = np.random.rand(m,k)
print ('W1 shape')
print W1.shape
H1 = np.random.rand(n,k)
print ('H1 shape')
print H1.shape

m,n = X2.shape
W2 = np.random.rand(m,k)
print ('W2 shape')
print W2.shape
H2 = np.random.rand(n,k)
print ('H2 shape')
print H2.shape

W1 = W1/10.0
H1 = H1/10.0

W2 = W2/10.0
H2 = H2/10.0

print ('W1')
print W1
print ('H1')
print H1

print ('W2')
print W2
print ('H2')
print H2

print ('----------')

index = []
errSqrX1 = []
errAbsX1 = []
errSqrX2 = []
errAbsX2 = []

for e in range(epoc):
    alpha = 0.01/np.sqrt(e+1)
    print e

    #Wn = W - alpha * ( reduce(np.dot, [W,np.transpose(H),H] ) - np.dot(X,H) + W )
    #Hn = H - alpha * ( reduce(np.dot, [H,np.transpose(W),W] ) - np.dot(np.transpose(X),W) + H )
    
    W1n = W1 - alpha * ( -2 * np.dot( (X1 - np.dot(W1, np.transpose(H1))) , H1 ) + 2 * W1 )
    H1n = H1 - alpha * ( -2 * np.dot( np.transpose( X1 - np.dot(W1 , np.transpose(H1)) ), W1 ) + 2 * H1 )
    
    W2n = W2 - alpha * ( -2 * np.dot( (X2 - np.dot(W2, np.transpose(H2))) , H2 ) + 2 * W2 )
    H2n = H2 - alpha * ( -2 * np.dot( np.transpose( X2 - np.dot(W2 , np.transpose(H2)) ), W2 ) + 2 * H2 )
    '''
    print ('W1n')
    print W1n
    print ('H1n')
    print H1n
    print ('W2n')
    print W2n
    print ('H2n')
    print H2n
    '''
    #print ('---------------------------------------------------------')
    
    errorSqrX1 = lossfuncSqr(X1,W1n,np.transpose(H1n))
    errorAbsX1 = lossfuncAbs(X1,W1n,np.transpose(H1n))
    errorSqrX2 = lossfuncSqr(X2,W2n,np.transpose(H2n))
    errorAbsX2 = lossfuncAbs(X2,W2n,np.transpose(H2n))
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

print np.dot(W1n,np.transpose(H1n))
print np.dot(W2n,np.transpose(H2n))


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