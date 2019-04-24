# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:18:15 2019

@author: mirza
"""

import numpy as np
from sklearn.cluster import KMeans 

def merge():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

    with open(path + 'W1c.txt') as file:
        arrayW1c = [[float(digit) for digit in line.split('\t')] for line in file]
        
    with open(path + 'W2c.txt') as file:
        arrayW2c = [[float(digit) for digit in line.split('\t')] for line in file]
    
    with open(path + 'W1d.txt') as file:
        arrayW1d = [[float(digit) for digit in line.split('\t')] for line in file]
    
    with open(path + 'W2d.txt') as file:
        arrayW2d = [[float(digit) for digit in line.split('\t')] for line in file]
        
    W1c = np.array(arrayW1c)
    W2c = np.array(arrayW2c)
    W1d = np.array(arrayW1d)
    W2d = np.array(arrayW2d)

    print W1c.shape
    
    #V = np.hstack(( (W1c + W2c) / 2 , np.hstack((W1d, W2d)) ))
    
    W1cW2c = (W1c + W2c) / 2
    np.savetxt(path + "W1cW2c.csv", W1cW2c, delimiter=",")
    
    W1dW2d = np.hstack(( W1d, W2d ))
    np.savetxt(path + "W1dW2d.csv", W1dW2d, delimiter=",")
    
    #np.savetxt(path + "vector.csv", V, delimiter=",")
    
def kmeans():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

    with open(path + 'vector.txt') as file:
        arrayV = [[float(digit) for digit in line.split('\t')] for line in file]
        
    ptrn = []
    with open(path + 'pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())
    
    print ptrn
    
    V = np.array(arrayV)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(V)
    
    fw = open(path + "VectorCluster3.txt", "w")
    '''
    for l in kmeans.labels_:
        print l
        fw.write(str(l)+"\n")
    '''
    for i in range (len(kmeans.labels_)):
        print ptrn[i], kmeans.labels_[i]
        fw.write(ptrn[i] + "\t" + str(kmeans.labels_[i]) + "\n")
    
if __name__ == "__main__":
    print ('Start:')
    #merge()
    kmeans()
    print ('End')