# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:26:17 2019

@author: mirza
"""
path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040nmft/'
path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/test/'

from sklearn.decomposition import NMF
import numpy as np

def nmf(X, k):
    model = NMF(n_components=k, init='random', random_state=None)
    W = model.fit_transform(X)
    H = model.components_
    e = model.reconstruction_err_
    return W,H,e
    
def iterate(X1, X2, k):
    W1, H1, e1 = nmf(X1, k)
    W2, H2, e2 = nmf(X2, k)
    
    pathk = path + "k" + str(k)
    print pathk
    
    np.savetxt(pathk + "W1.csv", W1, delimiter=",")
    np.savetxt(pathk + "H1.csv", H1, delimiter=",")
    
    np.savetxt(pathk + "W2.csv", W2, delimiter=",")
    np.savetxt(pathk + "H2.csv", H2, delimiter=",")
    
    fw1 = open(pathk + 'err1.txt', "w")
    fw2 = open(pathk + 'err2.txt', "w")
    
    fw1.write(str(e1))
    fw2.write(str(e2))
    
    fw1.close
    fw2.close

def merge_error_nmf(k):    
    for j in range (1,3):
        err = []
        for i in range(1, k+1):
            file_name = path + "k" + str(i) + "err" + str(j) + ".txt"
            f = open(file_name)
            line = f.readline()
            err.append(line.strip())
            
        fw = open(path + "/err" + str(j) + ".txt", "w")
        for i in err:
            fw.write(i + ",")
    
        fw.close()
            
    
    err = []
    for i in range(1, k+1):
        file_name = path + "k" + str(i) + "err1.txt"
        f = open(file_name)
        line = f.readline()
        err.append(line.strip())
    
    fw = open(path + '/err1.txt', "w")
    for i in err:
        fw.write(i + ",")
    
    fw.close()
    
    err = []
    for i in range(1, k+1):
        file_name = path + "k" + str(i) + "err2.txt"
        f = open(file_name)
        line = f.readline()
        err.append(line.strip())
    
    fw = open(path + '/err1.txt', "w")
    for i in err:
        fw.write(i + ",")
    
    fw.close()

if __name__ == "__main__":
    '''
    with open(path + 'l.txt') as file:
        array1 = [[float(digit) for digit in line.split('\t')] for line in file]        
    X1 = np.array(array1)
    
    with open(path + 'h.txt') as file:
        array2 = [[float(digit) for digit in line.split('\t')] for line in file]        
    X2 = np.array(array2)
    
    k = 40
    
    for i in range(1, k+1):
        iterate(X1, X2, i)
    
    merge_error_nmf(k)
    '''
    
    with open(path + 'l.txt') as file:
        array1 = [[float(digit) for digit in line.split('\t')] for line in file]        
    X1 = np.array(array1)
    
    W,H,e = nmf(X1, 20)
    print e
    
    
    