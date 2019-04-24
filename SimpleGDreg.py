# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 18:39:32 2019

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
    

epoc = 50
alpha = 0.01
k = 2

m,n = X.shape

print X.shape

W = np.random.rand(m,k)
H = np.random.rand(n,k)

#W = W/100.0
#H = H/100.0

print W
print H

print ('----------')

Wn = np.zeros((m,k))
Hn = np.zeros((n,k))

index = []
errSqr = []
errAbs = []

for e in range(epoc):
    alpha = 0.01/np.sqrt(e+1)

    #Wn = W - alpha * ( reduce(np.dot, [W,np.transpose(H),H] ) - np.dot(X,H) + W )
    #Hn = H - alpha * ( reduce(np.dot, [H,np.transpose(W),W] ) - np.dot(np.transpose(X),W) + H )
    
    Wn = W - alpha * ( -2 * np.dot( (X - np.dot(W, np.transpose(H))) , H ) + 2 * W )
    Hn = H - alpha * ( -2 * np.dot( np.transpose( X - np.dot(W , np.transpose(H)) ), W ) + 2 * H )
    
    #print ('---------------------------------------------------------')
    errorSqr = lossfuncSqr(X,Wn,np.transpose(Hn))
    errorAbs = lossfuncAbs(X,Wn,np.transpose(Hn))
    index.append(e)
    errSqr.append(errorSqr)
    errAbs.append(errorAbs)
    
    W = Wn
    H = Hn
    
    print ('Wn new')
    print Wn
    print ('Hn new')
    print Hn
    
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