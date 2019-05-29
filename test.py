# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:07:24 2019

@author: mirza
"""

import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def test_multiline():
    x = (2
         +3)
    
    print x

def test_multiplot():
    X = [1,2,3]
    Y = [4,5,6]
    
    plt.plot(X, label = "X")
    plt.plot(Y, label = "Y")
    plt.legend()
    plt.show

def test_matrix():
    X = np.array([[-1.0,  0.0,   3.0],
              [ 4.0,   -5.0,  6.0],
              [ 7.0,   -8.0,  9.0]])
    
    print X[1,1]
    print X[-1,-1]

def test_plot():
    X = [0.076735955479304,0.12107471428354363,0.15925099171396867,0.12117445698341163,0.15825396440460873,0.16551976695172887,0.15115833440529808,0.12975000378573398,0.15124189274020355,0.15876507845507717,0.14610753559819975,0.13430067074012042,0.12630876778360994,0.14173878577693938,0.12135526213786856,0.15310386171908613,0.14416107113263588,0.1421683239601366,0.13623913727189219,0.13899612212284504,0.12533236140227075,0.12800740495456878,0.12183144900467016,0.11682153035681135,0.12118286832735578,0.11998127152505512,0.11947487328137382,0.11850810984694396,0.11481165386753438,0.10757759056999501]
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(X, linewidth =2.0, color = 'red', ls='None', marker = '+')
    plt.show

def test_add_element():
    X = np.array([[-1.0,  0.0,   3.0],
              [ 4.0,   -5.0,  6.0],
              [ 7.0,   -8.0,  9.0]])
    
    X[1,1] += 0.1
    
    print X
    
    X[0,0] -= 2 * 0.1
    
    print X
    
def test_forb():
    X = np.array([[-1,  0,   3],
              [ 4,   -5,  6],
              [ 7,   -8,  9]])
    
    print np.dot(X, np.transpose(X)).flatten().sum()
    print math.pow(10,-4)
    print X[0,0]
     
    return np.linalg.norm(X, ord=None, axis=None, keepdims=False) ** 2

def test_nonzero():
    X = np.array([[-1,  0,   3],
              [ 4,   -5,  6],
              [ 7,   -8,  9]])
    m, n = X.shape
    Y = X
    for i in range(m):
        for j in range(n):
            if (X[i,j] > 0):
                Y[i,j] = X[i,j]
            else:
                Y[i,j] = 0
    print Y
    X[X<0] = 0
    print X

def test_auto():
    a = np.array([1,2,3,4])
    b = np.array([[-1,2],[3,4]])
    print a.mean()
    b[b<0] = 0
    print b

def test_row():
    for i in range (2,21):
        for j in range(1,i):
            print str(i) + "," + str(j)

def test_pow():
    print math.pow(6 - 3, 2)

def make_rand():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/test/"
    n = np.random.rand(100,100)
    n = n *100
    a, b = n.shape
    m = n
    for i in range(a):
        for j in range(b):
            m[i,j] = int(n[i,j])
    #print m
    
    n2 = np.random.rand(100,100)
    n2 = n2 *100
    a2, b2 = n2.shape
    m2 = n2
    for i in range(a2):
        for j in range(b2):
            m2[i,j] = int(n2[i,j])
    #print m2
    
    np.savetxt(path + "/htest.csv", m, delimiter=",")
    np.savetxt(path + "/ltest.csv", m2, delimiter=",")

def test_read_csv():
    pathnn = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/k10/c2d8/"
    
    f1 = open(pathnn + "W1c.csv")
    arrayW1c = [[float(digit) for digit in line.split(',')] for line in f1]
    
    print arrayW1c

def test_sum_array():
    x = np.array([[1,  2,   3],
              [ 4,   5,  6],
              [ 7,   8,  9]])
    
    y = np.array([[1,  2,   3]])
    
    z = np.zeros(3)
    
    print y + z

def test_stack():
    X = np.zeros(shape =(1,2))
    X [0,0] = 1
    X [0,1] = 2
    Y = np.zeros(shape =(1,2))
    Y [0,0] = 3
    Y [0,1] = 4
    Z = np.hstack((X,Y))
    print Z

def test_csv():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/k10/c2d8/"
    with open(path + 'W1.csv') as file:
        array2d = [[float(digit) for digit in line.split(',')] for line in file]
        
    X = np.array(array2d)
    print X.shape
    print X
        
def test_sum():
    x = np.array([[1,  2,   3],
              [ 4,   5,  6],
              [ 7,   8,  9]])
    y = np.array([[3,  2,   1],
              [ 6,   5,  4],
              [ 9,   8,  7]])
    
    z = (x + y) / 2
    print z
    
    w = np.hstack((x, y))
    print w
    
    V = np.hstack(( (x + y) / 2 , np.hstack((x, y)) ))
    print V
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(V)
    
    print(kmeans.labels_)
    for l in kmeans.labels_:
        print l

def test_divide():
    X = np.array([
        [50, 30, 0, 10, 40, 20],
        [40, 0, 0, 10, 30, 50],
        [10, 10, 0, 50, 10, 0],
        [10, 0, 0, 40, 20, 20],
        [0, 10, 50, 40, 30, 40],
    ])
    
    a = X[:, 2:]
    b = X[:, :2]
    
    print a.shape
    print a
    print b.shape
    print b
    

def test_sum_one():
    X = np.array([
        [50, 30, 0, 10],
        [40, 0, 0, 10],
        [10, 10, 0, 50],
        [10, 0, 0, 40],
        [0, 10, 50, 40],
    ])
    
    print X / X.max(axis = 0)
    print X / X.sum(axis = 0)
    return X / X.sum(axis = 0)

def copy():
    x = np.array([[1000,  10,   0.5],
              [ 765,   5,  0.35],
              [ 800,   7,  0.09]])
    X = np.array([
        [50.0, 30.0, 0.0, 10.0],
        [40.0, 0.0, 0.0, 10.0],
        [10.0, 10.0, 0.0, 50.0],
        [10.0, 0.0, 0.0, 40.0],
        [0.0, 10.0, 50.0, 40.0],
    ])
    x = X
    x_normed = x / x.max(axis=0)
    print (x_normed)
    x_normed1 = x / x.sum(axis=0)
    print (x_normed1)

def multiplying():

    X = np.array([
            [5, 3, 0, 1],
            [4, 0, 0, 1],
            [1, 1, 0, 5],
            [1, 0, 0, 4],
            [0, 1, 5, 4],
            ])

    X = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])

    A = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
    ])
    
    print A.shape
    
    B = np.array([
        [5, 3, 0, 1 ,2],
        [4, 0, 0, 1 ,3],
        [1, 1, 0, 5 ,1],
        [1, 0, 0, 4, 4],
    ])    
    
    wd = X[:,:2]
    print wd
    
    print X[:,2:]    
    
    hc = X[:2,:]
    print hc
    
if __name__ == "__main__":
    #test_sum()
    #test_divide()
    #test_csv()
    #test_stack()
    #test_sum_array()
    #test_read_csv()
    #make_rand()
    #test_pow()
    #test_row()
    #test_auto()
    #test_nonzero()
    #print test_forb()
    #test_add_element()
    #test_plot()
    #test_matrix()
    #test_multiplot()
    test_multiline()