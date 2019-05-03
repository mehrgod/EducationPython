# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:18:15 2019

@author: mirza
"""

import os
import numpy as np
from sklearn.cluster import KMeans 
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
#from sklearn.metrics import pairwise_distances

def merge_errors():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/k22/"
    errors = []
    for x in os.listdir(path):
        print x
        pathn = path + x
        with open(pathn + "/err.txt") as file:
            array2d = [line for line in file]
            errors.append(array2d[4])
            #errors.append(float("{0:.8f}".format(array2d[4])))
            
    #err = np.array(errors)
    
    #np.savetxt(path + "/errs.txt", err, delimiter=",")
    fw = open(path + "/errs.txt", "w")
    for e in errors:
        fw.write(e.strip()+",")
    fw.close()
            
    

def find_error():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/k10/c2d8/"
    with open(path + 'W1.csv') as file:
        array2d = [[float(digit) for digit in line.split(',')] for line in file]
        
      
    X = np.array(array2d)
    return X

def merge():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/k20/c15d5/'

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
    
    V = np.hstack(( (W1c + W2c) / 2 , np.hstack((W1d, W2d)) ))
    
    W1cW2c = (W1c + W2c) / 2
    np.savetxt(path + "W1cW2c.csv", W1cW2c, delimiter=",")
    
    W1dW2d = np.hstack(( W1d, W2d ))
    np.savetxt(path + "W1dW2d.csv", W1dW2d, delimiter=",")
    
    np.savetxt(path + "vector.csv", V, delimiter=",")

def test_centroid():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'
    with open(path + 'vector.txt') as file:
        arrayV = [[float(digit) for digit in line.split('\t')] for line in file]
        
    V = np.array(arrayV)
    kmeans = KMeans(n_clusters = 5, random_state = 0).fit(V)
    for c in kmeans.cluster_centers_:
        print c
    y_kmeans = kmeans.predict(V)
    plt.scatter(V[:, 0], V[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    
def kmeans():
    
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'
    
    ptrn = []
    with open(path + 'pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())
    
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/k20/c15d5/'

    with open(path + 'vector.txt') as file:
        arrayV = [[float(digit) for digit in line.split('\t')] for line in file]
        
    
    clusters = 2
    
    V = np.array(arrayV)
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(V)
    
    new_path = path + str(clusters)
    
    try:
        os.mkdir(new_path)
    except OSError:
        print ("Directory %s already exists!" %new_path)
    else:
        print ("Successfully created the directory %s " %new_path)
    
    fw = open(new_path + "/VectorCluster" + str(clusters) +".txt", "w")
    
    for i in range (len(kmeans.labels_)):
        print ptrn[i], kmeans.labels_[i]
        fw.write(ptrn[i] + "\t" + str(kmeans.labels_[i]) + "\n")
    
    fw_cent = open(new_path + "/VectorCenter" + str(clusters) +".txt", "w")
        
    for c in kmeans.cluster_centers_:
        fw_cent.write(str(c) + "\n")
        
    fw_score = open(new_path + "/ClusteringScore" + str(clusters) +".txt", "w")
    
    labels = kmeans.labels_
    print(metrics.silhouette_score(V, labels, metric='euclidean'))
    print(metrics.calinski_harabaz_score(V, labels))
    print(davies_bouldin_score(V, labels))
    fw_score.write("Silhouette Score: " + str(metrics.silhouette_score(V, labels, metric='euclidean')) + "\n")
    fw_score.write("Calinski Harabaz Score: " + str(metrics.calinski_harabaz_score(V, labels)) + "\n")
    fw_score.write("Davies Bouldin Score: " + str(davies_bouldin_score(V, labels)))
    
    file.close()
    fw.close()
    fw_cent.close()
    fw_score.close()
    
def match_id():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'
    
    idlist = []
    with open(path + 'IdLo.txt') as ids:
        for i in ids:
            idlist.append(i.strip())
    
    vecdict = {}
    with open(path + 'VectorNormal1.txt') as vecs:
        for line in vecs:
            (key, val) = line.split("\t")
            vecdict[key.strip()] = val.strip()
    
    for x in vecdict:
        if x in idlist:
            print x
            
    fw = open(path + "/VectorNormalLo.txt", "w")
    for x in vecdict:
        if x in idlist:
            fw.write(x + "\t" + vecdict[x] + "\n")
    
    fw.close
    
if __name__ == "__main__":
    print ('Start:')
    #merge()
    #kmeans()
    #test_centroid()
    #match_id()
    merge_errors()
    print ('End')