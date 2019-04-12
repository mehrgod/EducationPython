# -*- coding: utf-8 -*-
"""
Created on Tue Mar 05 20:42:31 2019

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


X1 = np.array([
    [5, 3, 0, 1, 2, 4],
    [4, 0, 0, 1 ,0, 0],
    [1, 1, 0, 5 ,3, 1],
    [1, 0, 0, 4 ,4, 2],
    [0, 1, 5, 4 ,1, 2],
])

X2 = np.array([
    [5, 3, 0, 1, 2, 4],
    [4, 0, 0, 1 ,0, 0],
    [1, 1, 0, 5 ,3, 1],
    [1, 0, 0, 4 ,4, 2],
    [0, 1, 5, 4 ,1, 2],
])

path = 'C:/Project/EDU/files/2013/example/Topic/60/'

with open(path + 'X1.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]

X1 = np.array(array2d)
X2 = np.array(array2d)

'''    
X2 = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])
'''
beta = 0

epoc = 20
alpha = 0.01
k = 4
kc = k / 2
kd = k / 2

m1,n1 = X1.shape
m2,n2 = X2.shape

W1 = np.random.rand(m1,k)
H1 = np.random.rand(n1,k)

W1 = W1/100.0
H1 = H1/100.0

P = np.random.rand(m1,m1)

#W1n = np.zeros((m1,k))
#H1n = np.zeros((k,n1))

W1c = W1[:,:k/2]
W1d = W1[:,k/2:]
W1cn = W1[:,:k/2]
W1dn = W1[:,k/2:]

H1n = np.random.rand(n1,k)
H1c = H1[:,:k/2]
H1d = H1[:,k/2:]
H1cn = H1[:,:k/2]
H1dn = H1[:,:k/2]



###########################

W2 = np.random.rand(m2,k)
H2 = np.random.rand(n2,k)

#W2n = np.zeros((m2,k))
#H2n = np.zeros((k,n2))

W2c = W2[:,:k/2]
W2d = W2[:,k/2:]
W2cn = W2[:,:k/2]
W2dn = W2[:,k/2:]

H2n = np.random.rand(n2,k)
H2c = H2[:,:k/2]
H2d = H2[:,k/2:]
H2cn = H2[:,:k/2]
H2dn = H2[:,:k/2:]

print ("W1c ", W1c.shape)
print ("W1d ", W1d.shape)
print ("X1 ", X1.shape)
print ("W1 ", W1.shape)
print ("X2 ", X2.shape)
print ("H1 ", H1.shape)
print ("H2 ", H2.shape)
print ("H1c ", H1c.shape)
print ("H2c ", H2c.shape)
print ("H2d ", H2d.shape)
print ("W2c ", W2c.shape)
print ("W2d ", W2d.shape)
print ("W2 ", W2.shape)
print ("P" , P.shape)


index = []
errSqr = []
errAbs = []


for e in range(epoc):
    alpha = 0.1/np.sqrt(e+1)
    print alpha
    '''
    W1cn = W1c - alpha * 2 * (np.dot((X1 - np.dot(W2,np.transpose(H1))), np.transpose(H1c)) + 2 * (W1c - W2c) + 2 * np.dot((np.dot(W1,np.transpose(W1))) , W1c))
    W2cn = W2c - alpha * 2 * (np.dot((X2 - np.dot(W1,np.transpose(H2))), np.transpose(H2c)) + 2 * (W1c - W2c) + 2 * np.dot((np.dot(W2,np.transpose(W2))) , W2c))
    
    W1dn = W1d - alpha * 2 * (np.dot((X1 - np.dot(W1,np.transpose(H1))), np.transpose(H1d)) + 2 * np.dot( np.dot(np.transpose(W1d),W2d) , W2d ) + 2 * np.dot(np.dot(W1, np.transpose(W1)) - P , W1d))   
    W2dn = W2d - alpha * 2 * (np.dot((X2 - np.dot(W2,np.transpose(H2))), np.transpose(H2d)) + 2 * np.dot( np.dot(np.transpose(W1d),W2d) , W1d ) + 2 * np.dot(np.dot(W2, np.transpose(W2)) - P , W2d))   
    
    H1cn = H1c + np.dot(np.transpose(W1), X1 - np.dot(W1,np.transpose(H1)))
    H2cn = H2c + np.dot(np.transpose(W2), X2 - np.dot(W2,np.transpose(H2)))
    
    #Wn = W + 2 * alpha * np.dot((X - np.dot(W,H)), np.transpose(H))
    #Hn = H + 2 * alpha * np.dot(np.transpose(W), (X - np.dot(W,H)))
    '''
    #np.dot((X1 - np.dot(W1,np.transpose(H1))), H1c)
    #W1c - W2c
    #np.dot((np.dot(W1,np.transpose(W1))) - P , W1c)
    W1cn = W1c - alpha * 2 * (-2 * np.dot((X1 - np.dot(W1,np.transpose(H1))), H1c) + 2 * (W1c - W2c) + 2 * beta * np.dot((np.dot(W1,np.transpose(W1))) - P , W1c))
    W2cn = W2c - alpha * 2 * (-2 * np.dot((X2 - np.dot(W2,np.transpose(H2))), H2c) - 2 * (W1c - W2c) + 2 * beta * np.dot((np.dot(W2,np.transpose(W2))) - P , W2c))
    
    W1dn = W1d - alpha * 2 * (-2 * np.dot((X1 - np.dot(W1,np.transpose(H1))), H1d) + 2 * np.dot( W2d, np.dot(np.transpose(W1d),W2d) ) + 2 * beta * np.dot(np.dot(W1, np.transpose(W1)) - P , W1d))
    '''
    np.dot((X2 - np.dot(W2,np.transpose(H2))), H2d)
    np.dot( W1d, np.dot(np.transpose(W1d),W2d) )
    np.dot(np.dot(W2, np.transpose(W2)) - P , W2d)
    '''
    W2dn = W2d - alpha * 2 * (-2 * np.dot((X2 - np.dot(W2,np.transpose(H2))), H2d) + 2 * np.dot( W1d, np.dot(np.transpose(W1d),W2d) ) + 2 * beta * np.dot(np.dot(W2, np.transpose(W2)) - P , W2d))   
    
    '''
    H1n = H1 - 2 * np.dot(np.transpose(W1), X1 - np.dot(W1,np.transpose(H1)))
    H2n = H2 - 2 * np.dot(np.transpose(W2), X2 - np.dot(W2,np.transpose(H2)))
    '''
    
    H1n = H1 - alpha * 2 * (-2 * np.dot( np.transpose(X1 - np.dot(W1,np.transpose(H1))),W1 ))
    H2n = H2 - alpha * 2 * (-2 * np.dot( np.transpose(X2 - np.dot(W2,np.transpose(H2))),W2 ))
    
    W1n = np.concatenate((W1c,W2c),axis = 1)
    H1n = np.concatenate((H1c,H2c),axis = 1)

    
    errorSqr = lossfuncSqr(X1,W1n,np.transpose(H1n))
    errorAbs = lossfuncAbs(X1,W1n,np.transpose(H1n))
    index.append(e)
    errSqr.append(errorSqr)
    errAbs.append(errorAbs)
    #print np.dot(Wn,Hn)    
    
    W1c = W1cn
    W2c = W2cn
    W1d = W1dn
    W2d = W2dn
    H1 = H1n
    H2 = H2n
    
    print W1c
    print('----------------')
    print W1d

W1n = np.concatenate((W1c,W2c),axis = 1)
H1n = np.concatenate((H1c,H2c),axis = 1)

#print W1n
#print H1n

Xn = np.dot(W1n, np.transpose(H1n))
print Xn.shape
print (Xn)


plt.figure(1)
plt.plot(index,errSqr)
plt.title('Square Error')
plt.xlabel('Iteration')
plt.figure(2)
plt.plot(index,errAbs)
plt.title('Absolute Error')
plt.xlabel('Iteration')
#print np.dot(Wn,Hn)

