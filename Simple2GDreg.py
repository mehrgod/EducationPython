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
    return math.sqrt(sum / e)
    
def lossfuncAbs(X,Xn):
    m,n = X.shape
    sum = 0
    e = 0.0
    for i in range(m):
        for j in range(n):
            sum += abs(X[i,j] - Xn[i,j])
            e += 1
    return sum / e

path = 'C:/Project/EDU/files/2013/example/Topic/60/'
path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040s3/'

def grad_check(X, W, H):
    
    eps = math.pow(10, -4)
    Wtemp = W
    Htemp = H

    grad_W = 2 * np.dot((np.dot(W, H) - X), np.transpose(H)) + 2 * W
    grad_H = 2 * np.dot(np.transpose(W), (np.dot(W, H) - X )) + 2 * H
    
    #Xt = np.transpose(X)
    #Wt = np.transpose(W)
    #Ht = np.transpose(H)
    
    #grad_W = 2 * (reduce(np.dot,[W,Ht,H]) - np.dot(X,H) + W )
    #grad_H = 2 * (reduce(np.dot,[H,Wt,W]) - np.dot(Xt,W) + H )
    
    print ('W')
    print grad_W[1,1]
    
    Wtemp[1,1] += eps
    checkW1 = np.linalg.norm(X - np.dot(Wtemp, H), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Wtemp, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H, ord=None, axis=None, keepdims=False) ** 2
    
    Wtemp[1,1] -= 2 * eps
    checkW2 = np.linalg.norm(X - np.dot(Wtemp, H), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Wtemp, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H, ord=None, axis=None, keepdims=False) ** 2
    
    print (checkW1 - checkW2)/(2 * eps)
    
    print ('H')
    print grad_H[1,1]
    
    Htemp[1,1] += eps
    checkH1 = np.linalg.norm(X - np.dot(W, Htemp), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Htemp, ord=None, axis=None, keepdims=False) ** 2
    
    Htemp[1,1] -= 2 * eps
    checkH2 = np.linalg.norm(X - np.dot(W, Htemp), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Htemp, ord=None, axis=None, keepdims=False) ** 2
    
    print (checkH1 - checkH2)/(2 * eps)
'''
def grad_check_transpose(X, W, H):
    
    eps = math.pow(10, -4)
    Wtemp = W
    Htemp = H
    
    Xt = np.transpose(X)
    Wt = np.transpose(W)
    Ht = np.transpose(H)
    
    grad_W = 2 * (reduce(np.dot,[W,Ht,H]) - np.dot(X,H) + W )
    grad_H = 2 * (reduce(np.dot,[H,Wt,W]) - np.dot(Xt,W) + H )
    
    print ('W')
    print grad_W[1,1]
    
    Wtemp[1,1] += eps
    checkW1 = np.linalg.norm(X - np.dot(Wtemp, H), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Wtemp, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H, ord=None, axis=None, keepdims=False) ** 2
    
    Wtemp[1,1] -= 2 * eps
    checkW2 = np.linalg.norm(X - np.dot(Wtemp, H), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Wtemp, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H, ord=None, axis=None, keepdims=False) ** 2
    
    print (checkW1 - checkW2)/(2 * eps)
    
    print ('H')
    print grad_H[1,1]
    
    Htemp[1,1] += eps
    checkH1 = np.linalg.norm(X - np.dot(W, Htemp), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Htemp, ord=None, axis=None, keepdims=False) ** 2
    
    Htemp[1,1] -= 2 * eps
    checkH2 = np.linalg.norm(X - np.dot(W, Htemp), ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(W, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(Htemp, ord=None, axis=None, keepdims=False) ** 2
    
    print (checkH1 - checkH2)/(2 * eps)
'''
'''
def gd_new_transpose(k):
    with open(path + 'h.txt') as file:
        array2d = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X = np.array(array2d)
    
    print ('X')
    print X
    
    epoc = 100
    beta = 1.0
    
    m,n = X.shape
    
    W = np.random.rand(m,k)
    H = np.random.rand(n,k)
    
    index = []
    errSqr = []
    errAbs = []
    
    for e in range(epoc):
        alpha = 0.001/np.sqrt(e+1)
    
        Wt = np.transpose(W)
        Ht = np.transpose(H)
        
        grad_check_transpose(X, W, H)
        
        Wn = W - alpha * (
                np.dot((np.dot(W, Ht) - X), H)  
                + 2 * beta * W 
                )
        Hn = H - alpha * (
                np.dot(Wt, (np.dot(W, Ht) - X ))
                + 2 * beta * Ht
                )

        print ('---------------------------------------------------------')
        
        errorSqr = lossfuncSqr(X, np.dot(Wn, Hn))
        errorAbs = lossfuncAbs(X, np.dot(Wn, Hn))
        
        #print errorSqr
        #print errorAbs
        
        print "-----"

        index.append(e)
        
        errSqr.append(errorSqr)
        errAbs.append(errorAbs)

        W = Wn
        H = Hn
        
            
        print('W')
        print(W)
        print('H')
        print(H)
        
        print e
        if (e % 10 == 0):
            print e
            
    #err = lossfuncAbs(X, np.dot(W, H))
    #pathk = path + str(k)
    #fw = open(pathk + 'err.txt', "w")
    #fw.write(str(err))
    #fw.close()
    
    print ('Xn: ')
    print np.dot(W,H)
        
    plt.figure(1)
    plt.plot(index,errAbs)
    plt.title('Absolute Error')
    plt.xlabel('Iteration')
    
    plt.figure(2)
    plt.plot(index,errSqr)
    plt.title('Square Error')
    plt.xlabel('Iteration')
     
    plt.show()
'''    
def gd_new(k):
    with open(path + 'h.txt') as file:
        array2d = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X = np.array(array2d)
    
    #X = X / 10.0
    
    print ('X')
    print X
    
    epoc = 1000
    beta = 1.0
    
    m,n = X.shape
    
    W = np.random.rand(m,k)
    H = np.random.rand(k,n)
    
    W = W / 10.0
    H = H / 10.0
    
    #W = W * 10.0
    #H = H * 10.0
    
    index = []
    errSqr = []
    errAbs = []
    
    for e in range(epoc):
        alpha = 1.0/np.sqrt(e+1)
    
        Wt = np.transpose(W)
        Ht = np.transpose(H)
        
        #grad_check(X, W, H)
        
        grad_w = 2 * np.dot((np.dot(W, H) - X), Ht) 
        + 2 * beta * W
        
        Wn = W - alpha * grad_w
        
        grad_h = 2 * np.dot(Wt, (np.dot(W, H) - X )) 
        + 2 * beta * H
        
        Hn = H - alpha * grad_h
        
        Wn[Wn<0] = 0
        Hn[Hn<0] = 0

        #print ('---------------------------------------------------------')
        
        errorSqr = lossfuncSqr(X, np.dot(Wn, Hn))
        errorAbs = lossfuncAbs(X, np.dot(Wn, Hn))
        '''
        print errorSqr
        print errorAbs
        
        print "-----"
        '''

        index.append(e)
        
        errSqr.append(errorSqr)
        errAbs.append(errorAbs)

        W = Wn
        H = Hn
        '''
        if e > 10:
            if (errAbs[e] > errAbs[e-1]):
                print "Stopped at " + str(e)
                break
        '''
        '''    
        print('W')
        print(W)
        print('H')
        print(H)
        
        print e
        if (e % 10 == 0):
            print e
        '''    
    err = lossfuncAbs(X, np.dot(W, H))
    pathk = path + str(k)
    fw = open(pathk + 'err.txt', "w")
    fw.write(str(err))
    fw.close()
    
    #print ('Xn: ')
    #print np.dot(W,H)
        
    plt.figure(1)
    plt.plot(index,errAbs)
    plt.title('Absolute Error')
    plt.xlabel('Iteration')
    
    plt.figure(2)
    plt.plot(index,errSqr)
    plt.title('Square Error')
    plt.xlabel('Iteration')
     
    plt.show()
    
def read_error(k):
    index = []
    err = []
    for i in range(1, k+1):
        index.append(i)
        f = open(path + str(i) + "err.txt")
        lines = f.readlines()
        for l in lines:
            err.append(float(l))
     
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.plot(index,err,color = 'r',ls='None',marker = '.')
    plt.show    
        
    fw = open(path + "err.txt", "w")
    for e in err:
        fw.write(str(e) + ",")
    fw.close

def gd():

    with open(path + 'h.txt') as file:
        array2d = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X1 = np.array(array2d)
    
    with open(path + 'l.txt') as file:
        array2d = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X2 = np.array(array2d)
    
    epoc = 4
    alpha = 0.001
    k = 50
    
    m,n1 = X1.shape
    m,n2 = X2.shape
    
    print m, n1
    print m, n2
    
    W1 = np.random.rand(m,k)
    H1 = np.random.rand(n1,k)
    
    print W1.shape
    print H1.shape
    
    W2 = np.random.rand(m,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    print ('----------')
    
    print ('X1')
    print (X1)
    
    index = []
    errSqrX1 = []
    errAbsX1 = []
    errSqrX2 = []
    errAbsX2 = []
    
    alpha = 0.01
    
    for e in range(epoc):
        alpha = 0.1/np.sqrt(e+1)
    
        W1t = np.transpose(W1)
        H1t = np.transpose(H1)
        
        #W2t = np.transpose(W2)
        #H2t = np.transpose(H2)
        
        W1n = W1 - alpha * (
                - 2 * np.dot( (X1 - np.dot(W1, H1t)) , H1 ) 
                + 2 * alpha * W1 
                )    
        H1n = H1 - alpha * (
                #-2 * np.dot( np.transpose( X1 - np.dot(W1 , H1t) ), W1 ) 
                - 2 * np.dot( W1t,( np.dot(W1, H1t) - X1 ) )
                + 2 * alpha * H1 
                )
        '''
        W2n = W2 - alpha * (
                -2 * np.dot( (X2 - np.dot(W2, H2t)) , H2 ) 
                + 2 * W2 
                )
        H2n = H2 - alpha * (
                -2 * np.dot( np.transpose( X2 - np.dot(W2 , H2t) ), W2 ) 
                + 2 * H2 
                )
        '''
        print ('---------------------------------------------------------')
        
        errorSqrX1 = lossfuncSqr(X1, np.dot(W1, H1t))
        errorAbsX1 = lossfuncAbs(X1, np.dot(W1, H1t))
        '''
        errorSqrX2 = lossfuncSqr(X2, np.dot(W2, H2t))
        errorAbsX2 = lossfuncAbs(X2, np.dot(W2, H2t))
        '''
        index.append(e)
        
        errSqrX1.append(errorSqrX1)
        errAbsX1.append(errorAbsX1)
        '''
        errSqrX2.append(errorSqrX2)
        errAbsX2.append(errorAbsX2)
        '''
        W1 = W1n
        H1 = H1n
        '''
        W2 = W2n
        H2 = H2n
        '''
        print('W1')
        print(W1)
        print('H1')
        print(H1)
        
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
    '''
    plt.figure(3)
    plt.plot(index,errSqrX2)
    plt.title('Square Error X2')
    plt.xlabel('Iteration')
    
    plt.figure(4)
    plt.plot(index,errAbsX2)
    plt.title('Absolute Error X2')
    plt.xlabel('Iteration')
    '''
    plt.show
    
    np.savetxt(path + "W1.csv", W1, delimiter=",")
    np.savetxt(path + "W2.csv", W2, delimiter=",")


if __name__ == "__main__":
    
    k = 30
    '''
    print ('K = 2')
    gd_new(2)
    print ('K = 6')
    gd_new(6)
    print ('K = 10')
    gd_new(10)
    print ('K = 14')
    gd_new(14)
    print ('K = 18')
    gd_new(18)
    print ('K = 22')
    gd_new(22)
    print ('K = 26')
    gd_new(26)
    '''
    
    
    for i in range(1,k+1):
        print ('K = ' + str(i))
        gd_new(i)
    read_error(k)
    
    #gd_new(10)
    #gd_new_transpose(10)