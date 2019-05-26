# -*- coding: utf-8 -*-
"""
Created on Fri May 24 15:32:05 2019

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

def lossfuncD(X,Xn):
    sum = 0
    e = 0.0
    Y = np.dot(X,Xn)
    m,n = Y.shape
    for i in range(m):
        for j in range(n):
            sum += abs(Y[i,j])
            e += 1
    return sum/e

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040model/'

def function(X1, W1, H1, X2, W2, H2, kc):
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
    
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
        
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    v = np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(W1, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H1, ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(W2, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H2, ord=None, axis=None, keepdims=False) ** 2
    
    print ('ALL: ')
    print (v)
    
    return v

def function_sep(X1, W1, H1, X2, W2, H2, kc):
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
    
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
        
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    X1X2 = np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2
    C = np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2
    D = np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2
    
    return X1X2, C , D

def function_X1X2(X1, W1, H1, X2, W2, H2, kc):
    H1t = np.transpose(H1)
    H2t = np.transpose(H2)
    
    X1X2 = np.linalg.norm(X1 - np.dot(W1, H1t), ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(X2 - np.dot(W2, H2t), ord=None, axis=None, keepdims=False) ** 2
    
    print ('X1X2')
    print (X1X2)
    return X1X2
    
def function_C(X1, W1, H1, X2, W2, H2, kc):
    W1c = W1[:,:kc]
    W2c = W2[:,:kc]
    
    C = np.linalg.norm(W1c - W2c, ord=None, axis=None, keepdims=False) ** 2
    
    print ('C')
    print (C)
    return C

def function_D(X1, W1, H1, X2, W2, H2, kc):
    W1d = W1[:,kc:]
    W2d = W2[:,kc:]
        
    D = np.linalg.norm(np.dot(np.transpose(W1d), W2d) , ord=None, axis=None, keepdims=False) ** 2
    
    print ('D')
    print (D)
    return D

def function_R(X1, W1, H1, X2, W2, H2, kc):
    R = np.linalg.norm(W1, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H1, ord=None, axis=None, keepdims=False) ** 2
    + np.linalg.norm(W2, ord=None, axis=None, keepdims=False) ** 2 
    + np.linalg.norm(H2, ord=None, axis=None, keepdims=False) ** 2
    
    print ('R')
    print (R)
    return R
    
def grad_check_W1c(X1, W1, H1, X2, W2, H2, kc, grad):
    
    eps = math.pow(10, -3)
    W1temp = W1
    
    print ('W1c')
    print grad[1,1]
    e = grad[1,1]
    
    W1temp[1,1] += eps
    val1 = function(X1, W1temp, H1, X2, W2, H2, kc)
    
    W1temp[1,1] -= 2 * eps
    val2 = function(X1, W1temp, H1, X2, W2, H2, kc)
    
    c = (val1 - val2)/(2 * eps)
    
    print c
    
    return e,c
    
def grad_check_W2c(X1, W1, H1, X2, W2, H2, kc, grad):
    
    eps = math.pow(10, -3)
    W2temp = W2
    
    print ('W2c')
    print grad[1,1]
    e = grad[1,1]
    
    W2temp[1,1] += eps
    val1 = function(X1, W1, H1, X2, W2temp, H2, kc)
    
    W2temp[1,1] -= 2 * eps
    val2 = function(X1, W1, H1, X2, W2temp, H2, kc)
    
    c = (val1 - val2)/(2 * eps)
    
    return e,c

def grad_check_W1d(X1, W1, H1, X2, W2, H2, kc, grad):
    
    eps = math.pow(10, -3)
    W1temp = W1
    
    print ('W1d')
    print grad[-1,-1]
    e = grad[-1,-1]
    
    W1temp[-1,-1] += eps
    val1 = function(X1, W1temp, H1, X2, W2, H2, kc)
    
    W1temp[-1,-1] -= 2 * eps
    val2 = function(X1, W1temp, H1, X2, W2, H2, kc)
    
    c = (val1 - val2)/(2 * eps)
    
    return e,c

def grad_check_W2d(X1, W1, H1, X2, W2, H2, kc, grad):
    
    eps = math.pow(10, -3)
    W2temp = W2
    
    print ('W2d')
    print grad[-1,-1]
    e = grad[-1,-1]
    
    W2temp[-1,-1] += eps
    val1 = function(X1, W1, H1, X2, W2temp, H2, kc)
    
    W2temp[-1,-1] -= 2 * eps
    val2 = function(X1, W1, H1, X2, W2temp, H2, kc)
    
    c = (val1 - val2)/(2 * eps)
    
    return e,c

def gd_new_full(k, kc):
    
    with open(path + 'l.txt') as file:
        array2dX1 = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X1 = np.array(array2dX1)
    
    with open(path + 'h.txt') as file:
        array2dX2 = [[float(digit) for digit in line.split('\t')] for line in file]
    
    X2 = np.array(array2dX2)
    
    print ('X1')
    print X1
    
    print ('X2')
    print X2
    
    epoc = 100
    beta = 1.0
    
    m1,n1 = X1.shape
    m2,n2 = X2.shape
    
    W1 = np.random.rand(m1,k)
    H1 = np.random.rand(n1,k)
    
    W2 = np.random.rand(m2,k)
    H2 = np.random.rand(n2,k)
    
    W1 = W2
    
    W1 = W1 / 10.0
    H1 = H1 / 10.0
    
    W2 = W2 / 10.0
    H2 = H2 / 10.0
    
    index = []
    errSqrX1 = []
    errAbsX1 = []
    errSqrX2 = []
    errAbsX2 = []
    errX1X2 = []
    errSqrC = []
    errAbsC = []
    errD = []
    
    total_err = []
    total_X1X2 = []
    total_C = []
    total_D = []
    total_R = []
    total_sep = []
    
    alpha = 0.4
    beta = 0.4
    #gama = 1.0 - (alpha + beta)
    gama = 0.2
    reg = 0.01
    '''
    W1cErr = []
    W1cChk = []
    W2cErr = []
    W2cChk = []
    W1dErr = []
    W1dChk = []
    W2dErr = []
    W2dChk = []
    '''
    for e in range(epoc):
        learning_rate = 1.0/np.sqrt(e+1)
    
        W1c = W1[:,:kc]
        W1d = W1[:,kc:]
        
        H1c = H1[:,:kc]
        H1d = H1[:,kc:]
        
        W2c = W2[:,:kc]
        W2d = W2[:,kc:]
        
        H2c = H2[:,:kc]
        H2d = H2[:,kc:]
        
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
                
        grad_w1c = 2 * gama * np.dot((np.dot(W1, H1t) - X1), H1c)
        + 2 * alpha * (W1c - W2c)
        + 2 * reg * W1c
        W1cn = W1c - learning_rate * grad_w1c
        
        grad_w2c = 2 * gama * np.dot((np.dot(W2, H2t) - X2), H2c)
        - 2 * alpha * (W1c - W2c)
        + 2 * reg * W2c
        W2cn = W2c - learning_rate * grad_w2c        
        
        grad_w1d = 2 * np.dot((np.dot(W1, H1t) - X1), H1d) 
        + 2 * beta * np.dot(W2d, np.dot(W1dt,W2d))
        + 2 * reg * W1d        
        W1dn = W1d - learning_rate * grad_w1d
        
        grad_w2d = 2 * np.dot((np.dot(W2, H2t) - X2), H2d) 
        + 2 * beta * np.dot(W1d, np.dot(W1dt,W2d))
        + 2 * reg * W2d
        W2dn = W2d - learning_rate * grad_w2d
        
        #grad_h1 = 2 * np.dot(W1t, (np.dot(W1, H1t) - X1)) 
        #+ 2 * beta * H1
        grad_h1 = -2 * gama * np.dot(np.transpose(X1 - np.dot(W1, H1t)), W1) + 2 * reg * H1
        H1n = H1 - learning_rate * grad_h1
        
        #grad_h2 = 2 * np.dot(W2t, (np.dot(W2, H2t) - X2)) 
        #+ 2 * beta * H2
        grad_h2 = -2 * gama * np.dot(np.transpose(X2 - np.dot(W2, H2t)), W2) + 2 * reg * H2
        H2n = H2 - learning_rate * grad_h2
        
        #(W1ce,W1cc) = grad_check_W1c(X1, W1, H1, X2, W2, H2, kc, grad_w1c)
        #W1cErr.append(W1ce)
        #W1cChk.append(W1cc)
        
        #(W2ce,W2cc) = grad_check_W2c(X1, W1, H1, X2, W2, H2, kc, grad_w2c)
        #W2cErr.append(W2ce)
        #W2cChk.append(W2cc)
        
        #(W1de,W1dc) = grad_check_W1d(X1, W1, H1, X2, W2, H2, kc, grad_w1d)
        #W1dErr.append(W1de)
        #W1dChk.append(W1dc)
        
        #(W2de,W2dc) = grad_check_W2d(X1, W1, H1, X2, W2, H2, kc, grad_w2d)
        #W2dErr.append(W2de)
        #W2dChk.append(W2dc)
        
        tot_err = function(X1, W1, H1, X2, W2, H2, kc)
        total_err.append(tot_err)
        
        tot_X1X2 = function_X1X2(X1, W1, H1, X2, W2, H2, kc)
        total_X1X2.append(tot_X1X2)
        
        tot_C = function_C(X1, W1, H1, X2, W2, H2, kc)
        total_C.append(tot_C)
        
        tot_D = function_D(X1, W1, H1, X2, W2, H2, kc)
        total_D.append(tot_D)
        
        tot_R = function_R(X1, W1, H1, X2, W2, H2, kc)
        total_R.append(tot_R)
        
        
        #sep_err = function_sep(X1, W1, H1, X2, W2, H2, kc)
        #total_sep.append(sep_err)
        
        W1n = np.concatenate((W1cn,W1dn),axis = 1)
        W2n = np.concatenate((W2cn,W2dn),axis = 1)
        
        W1n[W1n<0] = 0
        H1n[H1n<0] = 0

        W2n[W2n<0] = 0
        H2n[H2n<0] = 0

        #print ('---------------------------------------------------------')
        
        errorSqrX1 = lossfuncSqr(X1, np.dot(W1n, np.transpose(H1n)) )
        errorAbsX1 = lossfuncAbs(X1, np.dot(W1n, np.transpose(H1n)) )
        
        errorSqrX2 = lossfuncSqr(X2, np.dot(W2n, np.transpose(H2n)))
        errorAbsX2 = lossfuncAbs(X2, np.dot(W2n, np.transpose(H2n)))
        
        errorAbsC = lossfuncAbs(W1cn, W2cn)
        errorSqrC = lossfuncSqr(W1cn, W2cn)
        
        errorD = lossfuncD(np.transpose(W1dn), W2dn)
        
        '''
        print errorSqrX1
        print errorAbsX1
        print errorSqrX2
        print errorAbsX2
        
        print "-----"
        '''

        index.append(e)
        
        errSqrX1.append(errorSqrX1)
        errAbsX1.append(errorAbsX1)

        errSqrX2.append(errorSqrX2)
        errAbsX2.append(errorAbsX2)
        
        errX1X2.append((errorAbsX1 + errorAbsX2)/2)
        
        errAbsC.append(errorAbsC)
        errSqrC.append(errorSqrC)
        
        errD.append(errorD)

        W1 = W1n
        H1 = H1n
        
        W2 = W2n
        H2 = H2n
        
        '''
        if e > 10:
            if (errAbs[e] > errAbs[e-1]):
                print "Stopped at " + str(e)
                break
        '''
        
        '''    
        print('W1c')
        print(W1c)
        print('W2c')
        print(W2c)
        '''
        #print e
        if (e % 10 == 0):
            print e
    
    pathk = path + str(k)
    
    err1 = lossfuncAbs(X1, np.dot(W1, H1t))
    fw1 = open(pathk + 'err1.txt', "w")
    fw1.write(str(err1))
    fw1.close()
    
    err2 = lossfuncAbs(X2, np.dot(W2, H2t))
    fw2 = open(pathk + 'err2.txt', "w")
    fw2.write(str(err2))
    fw2.close()
    '''
    fw1c = open(path + '')
    
    fw2c = open(path + '')
    
    fw1d = open(path + '')
    
    fw2d = open(path + '')
    '''
    W1c = W1[:,:kc]
    W1d = W1[:,kc:]
        
    H1c = H1[:,:kc]
    H1d = H1[:,kc:]
        
    W2c = W2[:,:kc]
    W2d = W2[:,kc:]
        
    H2c = H2[:,:kc]
    H2d = H2[:,kc:]
    
    np.savetxt(path + "/W1c.csv", W1c, delimiter=",")
    np.savetxt(path + "/W1d.csv", W1d, delimiter=",")
        
    np.savetxt(path + "/H1c.csv", H1c, delimiter=",")
    np.savetxt(path + "/H1d.csv", H1d, delimiter=",")
    
    np.savetxt(path + "/W2c.csv", W2c, delimiter=",")
    np.savetxt(path + "/W2d.csv", W2d, delimiter=",")
        
    np.savetxt(path + "/H2c.csv", H2c, delimiter=",")
    np.savetxt(path + "/H2d.csv", H2d, delimiter=",")
    
    np.savetxt(path + "/W1.csv", W1, delimiter=",")
    np.savetxt(path + "/W2.csv", W2, delimiter=",")
        
    np.savetxt(path + "/H1.csv", H1, delimiter=",")
    np.savetxt(path + "/H2.csv", H2, delimiter=",")
        
    plt.figure(1)
    plt.plot(index,errAbsX1)
    plt.title('Absolute Error X1')
    plt.xlabel('Iteration')
    
    plt.figure(2)
    plt.plot(index,errSqrX1)
    plt.title('Square Error X1')
    plt.xlabel('Iteration')
     
    plt.figure(3)
    plt.plot(index,errAbsX2)
    plt.title('Absolute Error X2')
    plt.xlabel('Iteration')
    
    plt.figure(4)
    plt.plot(index,errSqrX2)
    plt.title('Square Error X2')
    plt.xlabel('Iteration')
     
    plt.figure(5)
    plt.plot(index,errAbsC)
    plt.title('Absolute Error C')
    plt.xlabel('Iteration')
    
    plt.figure(6)
    plt.plot(index,errSqrC)
    plt.title('Square Error C')
    plt.xlabel('Iteration')
    
    plt.figure(7)
    plt.plot(index,total_err)
    plt.title('Total Objective Funcion Error')
    plt.xlabel('Iteration')
    
    plt.figure(8)
    #plt.plot(index, total_err, label = 'Total')
    plt.plot(index, total_X1X2, label = 'X1X2')
    plt.plot(index, total_C, label = 'C')
    plt.plot(index, total_D, label = 'D')
    plt.plot(index, total_R, label = 'R')
    plt.legend()
    plt.title('All Errors')
    plt.xlabel('Iteration')
     
    plt.show()

    
def read_error(k,n):
    index = []
    err = []
    for i in range(1, k+1):
        index.append(i)
        f = open(path + str(i) + "err" + n + ".txt")
        lines = f.readlines()
        for l in lines:
            err.append(float(l))
     
    fig, ax = plt.subplots(figsize=(10, 5))
    
    plt.plot(index,err,color = 'r',ls='None',marker = '.')
    plt.show    
        
    fw = open(path + "err" + n + ".txt", "w")
    for e in err:
        fw.write(str(e) + ",")
    fw.close


if __name__ == "__main__":
    
    
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
    '''
    k = 30
    for i in range(1,k+1):
        print ('K = ' + str(i))
        gd_new_full(i)
    read_error(k,"1")
    read_error(k,"2")
    '''
    gd_new_full(10,5)