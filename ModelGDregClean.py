# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:26:46 2019

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

def lossfuncD(X,Xn):
    sum = 0
    e = 0.0
    Y = np.dot(X,Xn)
    m,n = Y.shape
    for i in range(m):
        for j in range(n):
            sum += Y[i,j]
            e += 1
    return sum/e
    
def nonzero(X):
    m, n = X.shape
    Y = X
    for i in range(m):
        for j in range(n):
            if (X[i,j] > 0):
                Y[i,j] = X[i,j]
            else:
                Y[i,j] = 0
    return Y

def normal_column(X):
    return X / X.sum(axis = 0)

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

with open(path + 'l.txt') as file:
    array1 = [[float(digit) for digit in line.split('\t')] for line in file]

with open(path + 'h.txt') as file:
    array2 = [[float(digit) for digit in line.split('\t')] for line in file]

#with open(path + 'p.txt') as file:
#    arrayp = [[float(digit) for digit in line.split('\t')] for line in file]

X1 = np.array(array1)
X2 = np.array(array2)
#P = np.array(arrayp)

#print P.max()
#P = (P.max() - P) / P.max()
#print P

print X1.shape

print ('X1')
print (X1)
print ('X2')
print (X2)

epoc = 100
#a = 0.01

k = 10

kc = k / 2
kd = k / 2

m,n1 = X1.shape
m,n2 = X2.shape

W1 = np.random.rand(m,k)
H1 = np.random.rand(n1,k)
#P = np.random.rand(m,m)
#P = np.zeros((m,m))

W1 = W1 / 10.0
H1 = H1 / 10.0

print ('W1')
print W1
print ('H1')
print H1

###########################

W2 = np.random.rand(m,k)
H2 = np.random.rand(n2,k)

W2 = W2 / 10.0
H2 = H2 / 10.0

'''
print ('W2')
print W2
print ('H2')
print H2
'''

alpha = 1.0
beta = 1.0
gama = 1.0

index = []

errSqrX1 = []
errAbsX1 = []

errSqrX2 = []
errAbsX2 = []

errSqrC = []
errAbsC = []

errD = []

errSqrP = []
errAbsP = []

for e in range(epoc):
    a = 0.01/np.sqrt(e+1)
    print a
    
    W1c =  W1[:,:kc]
    W1d =  W1[:,kd:]

    H1c =  H1[:,:kc]
    H1d =  H1[:,kd:]
    
    W2c =  W2[:,:kc]
    W2d =  W2[:,kd:]
    
    H2c =  H2[:,:kc]
    H2d =  H2[:,kd:]
    
    W1t = np.transpose(W1)
    H1t = np.transpose(H1)
    
    W2t = np.transpose(W2)
    H2t = np.transpose(H2)
    
    W1ct = np.transpose(W1c)
    W1dt = np.transpose(W1d)
    
    W2ct = np.transpose(W2c)
    W2dt = np.transpose(W2d)
    
    H1ct = np.transpose(H1c)
    H1dt = np.transpose(H1d)
    
    H2ct = np.transpose(H2c)
    H2dt = np.transpose(H2d)
        
    W1cn = W1c - a * (
            - 2 * np.dot((X1 - np.dot(W1, H1t)), H1c ) 
            + 2 * alpha * (W1c - W2c) 
            #+ 2 * gama * (np.dot((np.dot(P,W1c) + np.dot(Pt,W1c)), (reduce(np.dot, [W1ct,P,W1c] ) - reduce(np.dot, [W2ct,P,W2c] )))) 
            #+ 2 * gama * np.dot( (np.dot(W1,W1t) - P ) , W1c )
            + 2 * W1c 
            )
        
    W2cn = W2c - a * (
            - 2 * np.dot((X2 - np.dot(W2, H2t)), H2c ) 
            - 2 * alpha * (W1c - W2c) 
            #- 2 * gama * (np.dot((np.dot(P,W2c) + np.dot(Pt,W2c)), (reduce(np.dot, [W1ct,P,W1c] ) - reduce(np.dot, [W2ct,P,W2c] )))) 
            #+ 2 * gama * np.dot( (np.dot(W2,W2t) - P ) , W2c )
            + 2 * W2c 
            )
    
    W1dn = W1d - a * (
            - 2 * np.dot((X1 - np.dot(W1,H1t)), H1d )
            + 2 * beta * np.dot(W2d, np.dot(W1dt, W2d))
            #+ 2 * gama * ( np.dot( ( np.dot(P,W1d) + np.dot(Pt,W1d) ), ( (reduce(np.dot, [W1dt, P, W1d] )) - (reduce(np.dot, [W2dt, P, W2d] ))) ) ) 
            #+ 2 * gama * np.dot((np.dot(W1,W1t) - P) , W1d)
            + 2 * W1d
            )
    
    W2dn = W2d - a * (
            - 2 * np.dot((X2 - np.dot(W2,H2t)), H2d )
            + 2 * beta * np.dot(W1d, np.dot(W1dt, W2d))
            #- 2 * gama * ( np.dot( ( np.dot(P,W1d) + np.dot(Pt,W1d) ), ( (reduce(np.dot, [W1dt, P, W1d] )) - (reduce(np.dot, [W2dt, P, W2d] ))))) 
            #+ 2 * gama * np.dot((np.dot(W2,W2t) - P) , W2d)
            + 2 * W2d
            )
    
    H1n = H1 - a * (-2 * np.dot(np.transpose(X1 - np.dot(W1, H1t)), W1) + 2 * H1)
    H1n = nonzero(H1n)
    #H1n = normal_column(H1n)
    
    H2n = H2 - a * (-2 * np.dot(np.transpose(X2 - np.dot(W2, H2t)), W2) + 2 * H2)
    H2n = nonzero(H2n)
    #H2n = normal_column(H2n)
    
    W1n = np.concatenate((W1cn,W1dn),axis = 1)
    W1n = nonzero(W1n)
    #W1n = normal_column(W1n)
    
    W2n = np.concatenate((W2cn,W2dn),axis = 1)
    W2n = nonzero(W2n)
    #W2n = normal_column(W2n)
    '''
    print ('W1 New')
    print W1n
    print ('H1 New')
    print H1n
    
    print ('W2 New')
    print W2n
    print ('H2 New')
    print H2n
    '''
    errorSqrX1 = lossfuncSqr(X1, np.dot(W1n,np.transpose(H1n)))
    errorAbsX1 = lossfuncAbs(X1, np.dot(W1n,np.transpose(H1n)))
    
    errorSqrX2 = lossfuncSqr(X2, np.dot(W2n,np.transpose(H2n)))
    errorAbsX2 = lossfuncAbs(X2, np.dot(W2n,np.transpose(H2n)))
    
    errorSqrC = lossfuncSqr(W1cn, W2cn)
    
    errorAbsC = lossfuncAbs(W1cn, W2cn)
    
    errorD = lossfuncD(np.transpose(W1dn), W2dn)
    
    #errorSqrP = lossfuncSqr(reduce(np.dot, [W1t, P, W1] ), reduce(np.dot, [W2t, P, W2]))
    #errorAbsP = lossfuncAbs(reduce(np.dot, [W1t, P, W1] ), reduce(np.dot, [W2t, P, W2]))    
    
    
    index.append(e)
    errSqrX1.append(errorSqrX1)
    errAbsX1.append(errorAbsX1)
    
    errSqrX2.append(errorSqrX2)
    errAbsX2.append(errorAbsX2)
    errSqrC.append(errorSqrC)
    errAbsC.append(errorAbsC)
    errD.append(errorD)
    #errSqrP.append(errorSqrP)
    #errAbsP.append(errorAbsP)
    
    W1 = W1n    
    W2 = W2n    
    H1 = H1n    
    H2 = H2n
''' 
print ('Final W1')
print (W1)
print ('Final H1')
print (H1)

print ('Final W2')
print (W2)
print ('Final H2')
print (H2)
'''
'''    
Xn = np.dot(W2n, np.transpose(H2n))
print (Xn)
'''

np.savetxt(path + "W1.csv", W1, delimiter=",")
np.savetxt(path + "W2.csv", W2, delimiter=",")

#print ('W1d')
#print W1d

#print ('W2d')
#print W2d
#print W1c
#print W2c

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

plt.figure(5)
plt.plot(index,errSqrC)
plt.title('Square Error Common')
plt.xlabel('Iteration')

plt.figure(6)
plt.plot(index,errAbsC)
plt.title('Absolute Error Common')
plt.xlabel('Iteration')

plt.figure(7)
plt.plot(index,errD)
plt.title('Square Error Discriminative')
plt.xlabel('Iteration')

#plt.figure(8)
#plt.plot(index,errAbsP)
#plt.title('Absolute Error Pattern')
#plt.xlabel('Iteration')

#plt.figure(9)
#plt.plot(index,errSqrP)
#plt.title('Square Error Pattern')
#plt.xlabel('Iteration')

plt.show
