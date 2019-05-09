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
import math

def plot_error_c():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/"
    
    f = open(path + "C2-4to20.txt")
    line = f.readlines()
    
    kkc = [token.strip() for token in line[0].split('\t')]
    mean = [float(token.strip()) for token in line[1].split('\t')]
    stdv = [float(token.strip()) for token in line[2].split('\t')]
    
    bar = []
    for i in range(len(mean)):
        bar.append(1.96 * float(stdv[i]) / math.sqrt(10))
    
    
    plt.rcParams.update({'font.size': 14})
    plt.figure()

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.errorbar(kkc, mean, stdv, 
                 #linestyle='None', 
                 marker='+', 
                 linewidth=3.0)

    plt.legend(('X1','X2'))

    plt.show()

def merge_errors_c(k):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/10/k" + str(k) + "/"
    errorsX1 = []
    for x in os.listdir(path):
        if x.startswith("c") and ("c0" not in x):
            print x
            pathn = path + x
            with open(pathn + "/errC2.txt") as file:
                for line in file:
                    errorsX1.append(line)
                
    fw1 = open(path + "/errsC2.txt", "w")
    for e in errorsX1:
        fw1.write(e.strip()+",")
    
    fw1.close()
    
def take_avg_c(value):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/"
    
    error = []
    
    for iteration in range(1, 11):
        pathn = path + str(iteration)
        for k in os.listdir(pathn):
            if k.startswith("k" + value):
                pathnn = pathn + "/" + k
                with open(pathnn + "/errsC2.txt") as X1:
                    for line in X1:
                        error.append(line)
                                
    fw = open(path + 'k' + value + 'errC2.txt', "w")
    
    for i in error:
        fw.write(i + "\n")
      
    fw.close
    
def find_error_c():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/10/"
    for k in os.listdir(path):
            if k.startswith("k"):
                pathn = path + k + "/"
                for l in os.listdir(pathn):
                    if ((l.startswith("c")) and ("c0" not in l)):
                        pathnn = pathn + l + "/"
                        
                        f1 = open(pathnn + "W1c.csv")
                        arrayW1c = [[float(digit) for digit in line.split(',')] for line in f1]
                        W1c = np.array(arrayW1c)
                        
                        f2 = open(pathnn + "W2c.csv")
                        arrayW2c = [[float(digit) for digit in line.split(',')] for line in f2]
                        W2c = np.array(arrayW2c)
                                                
                        X = abs(W1c - W2c)
                        
                        m,n = X.shape
                        sum = 0
                        e = 0.0
                        for i in range(m):
                            for j in range(n):
                                sum += X[i,j]
                                e += 1
                        err = sum / e
                        
                        fw = open(pathnn + '/errC2.txt', "w")
                        fw.write(str(err))
    
    f1.close
    f2.close
    fw.close

def plot_error():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/"
    
    f1 = open(path + "X1-4to20.txt")
    line1 = f1.readlines()
    
    kkc1 = [token.strip() for token in line1[0].split('\t')]
    mean1 = [float(token.strip()) for token in line1[1].split('\t')]
    stdv1 = [float(token.strip()) for token in line1[2].split('\t')]
    #number1 = [int(token.strip()) for token in line1[3].split('\t')]
    
    bar1 = []
    for i in range(len(mean1)):
        bar1.append(1.96 * float(stdv1[i]) / math.sqrt(10))
    
    f2 = open(path + "X2-4to20.txt")
    line2 = f2.readlines()
    
    kkc2 = [token.strip() for token in line2[0].split('\t')]
    mean2 = [float(token.strip()) for token in line2[1].split('\t')]
    stdv2 = [float(token.strip()) for token in line2[2].split('\t')]
    #number2 = [int(token.strip()) for token in line2[3].split('\t')]
    
    bar2 = []
    for i in range(len(mean2)):
        bar2.append(1.96 * float(stdv2[i]) / math.sqrt(10))
    
    
    plt.rcParams.update({'font.size': 14})
    plt.figure()

    fig, ax = plt.subplots(figsize=(20, 10))

    plt.errorbar(kkc1, mean1, stdv1, 
                 #linestyle='None', 
                 marker='+', 
                 linewidth=3.0)

    plt.errorbar(kkc2, mean2, stdv2, 
                 #linestyle='None', 
                 marker='^', 
                 linewidth=3.0)

    plt.legend(('X1','X2'))

    plt.show()
    


def take_avg(value):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/"
    
    errorX1 = []
    errorX2 = []
    errorX1X2 = []
    
    #k
    #value = "4"
    
    for iteration in range(1, 11):
        pathn = path + str(iteration)
        for k in os.listdir(pathn):
            if k.startswith("k" + value):
                pathnn = pathn + "/" + k
                        
                with open(pathnn + "/errsX1.txt") as X1:
                    for line in X1:
                        errorX1.append(line)
                
                with open(pathnn + "/errsX2.txt") as X2:
                    for line in X2:
                        errorX2.append(line)
                
                with open(pathnn + "/errsX1X2.txt") as X1X2:
                    for line in X1X2:
                        errorX1X2.append(line)
    
    fw1 = open(path + 'k' + value + 'errX1.txt', "w")
    fw2 = open(path + 'k' + value + 'errX2.txt', "w")
    fw12 = open(path + 'k' + value + 'errX1X2.txt', "w")
    
    for i in errorX1:
        fw1.write(i + "\n")
    
    for i in errorX2:
        fw2.write(i + "\n")
    
    for i in errorX1X2:
        fw12.write(i + "\n")
        
    fw1.close
    fw2.close
    fw12.close
    

def merge_errors(k):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/10/k" + str(k) + "/"
    errorsX1 = []
    errorsX2 = []
    errorsX1X2 = []
    for x in os.listdir(path):
        if x.startswith("c"):
            print x
            pathn = path + x
            with open(pathn + "/err.txt") as file:
                array2d = [line for line in file]
                errorsX1.append(array2d[1])
                errorsX2.append(array2d[3])
                errorsX1X2.append(array2d[4])
                #errors.append(float("{0:.8f}".format(array2d[4])))
            
    #err = np.array(errors)
    
    #np.savetxt(path + "/errs.txt", err, delimiter=",")
    fw1 = open(path + "/errsX1.txt", "w")
    for e in errorsX1:
        fw1.write(e.strip()+",")
    
    fw2 = open(path + "/errsX2.txt", "w")
    for e in errorsX2:
        fw2.write(e.strip()+",")
    
    fw12 = open(path + "/errsX1X2.txt", "w")
    for e in errorsX1X2:
        fw12.write(e.strip()+",")
    
    fw1.close()
    fw2.close()
    fw12.close()

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
    #take_avg()
    #plot_error()
    #for i in range (4, 21, 2):
        #print i
    #    merge_errors_c(i)
    #merge_errors(5)
    #for i in range(4, 21, 2):
    #    take_avg_c(str(i))
    #find_error_c()
    #for i in range(4, 21, 2):
    #    take_avg_c(str(i))
    plot_error_c()
    print ('End')