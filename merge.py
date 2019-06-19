# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:18:15 2019

@author: mirza
"""

import os
import numpy as np
from sklearn.cluster import KMeans 
#from sklearn import metrics
#from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import math
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
import csv

    
def plot_all():
        path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/"
        
        k = 30
        
        fname = path + 'k' + str(k) + '/' + str(k) + '.txt'
        print fname
        f = open(fname)
        err = [line.strip() for line in f]
        X = [float(token.strip()) for token in err[0].split(',')]
        C = [float(token.strip()) for token in err[1].split(',')]
        D = [float(token.strip()) for token in err[2].split(',')]
        
        index = []
        for j in range(1,k):
            index.append(j)
        
        plt.figure(1)
        plt.plot(index,X)
        #plt.savefig(path + "C1" + str(k) + ".png")
        
        plt.figure(2)
        plt.plot(index,C)
        #fig.savefig(path + "C1" + performance + ".pdf")
        
        plt.figure(3)
        plt.plot(index,D)
        #fig.savefig(path + "C1" + performance + ".pdf")

        
def merge_err():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040fix/"
    
    for k in range(2,31):
        X = []
        C = []
        D = []
        for kc in range(1,k):
            fname = path + 'k' + str(k) + '/c' + str(kc) + 'd' + str(k-kc) + '/err.txt'
            print fname
            f = open(fname)
            err = [line.strip() for line in f]
            
            print err
            
            X.append(err[4])
            C.append(err[5])
            D.append(err[6])
        
        x = ''
        c = ''
        d = ''
        for i in range(len(X)):
            x = x + ',' + X[i]
            c = c + ',' + C[i]
            d = d + ',' + D[i]
        fw = open(path + 'k' + str(k) + '/' + str(k) + '.txt', "w")
        fw.write(x[1:] + '\n' + c[1:] + '\n' + d[1:])
        fw.close

def plot_err():
    index = [1,2,3,4,5,6,7,8,9,10]
    errX1X2 = [0.004451450313779434,0.004334864872754272,0.004424388894019901,0.004434751511609728,0.004374787297086295,0.004306767716273716,0.004263029506293,0.00451907891580362,0.004262061485957635,0.004324369733329577]
    errC = [0.0037704143185466814,0.004309607397247974,0.003942931254609225,0.004880068446524691,0.0035027154870666795,0.00332567422698915,0.0035147815540130366,0.003419251877509466,0.003286636068537827,0.003158807224574146]
    errD = [0.03809298320597188,0.03661334287445711,0.032561708283282906,0.034603884817708284,0.043424168777963526,0.046455482748096036,0.060513235253845325,0.03972313899469962,0.035354531406567635,0.004602924642988289]
    
    plt.figure(1)
    plt.title('X1X2')
    plt.plot(index, errX1X2)
    plt.figure(2)
    plt.title('C')
    plt.plot(index, errC)
    plt.figure(3)
    plt.title('D')
    plt.plot(index, errD)
    plt.show()

def plot_error_no_errorbar():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/"
    
    f12 = open(path + "AllplotX1X2.txt")
    line12 = f12.readlines()
    
    kkc12 = [token.strip() for token in line12[0].split('\t')]
    val12 = [float(token.strip()) for token in line12[1].split(',')]
    '''
    fc = open(path + "AllplotC.txt")
    lineC = fc.readlines()
    
    kkcC = [token.strip() for token in lineC[0].split('\t')]
    valC = [float(token.strip()) for token in lineC[1].split(',')]
    
    fd = open(path + "AllplotD.txt")
    lineD = fd.readlines()
    
    kkcD = [token.strip() for token in lineD[0].split('\t')]
    valD = [float(token.strip()) for token in lineD[1].split(',')]
    '''
    #plt.rcParams.update({'font.size': 14})
    #plt.figure()

    plt.subplots(figsize=(50, 10))

    plt.errorbar( kkc12, val12
                 #linestyle='None', 
                 #marker='+', 
                 #linewidth=3.0
                 )
    
    #plt.errorbar(kkcC, valC)
    
    #plt.errorbar(kkcD, valD)
    #plt.legend(('X1X2','C','D'))
    plt.legend(('X1X2'))
    
    plt.savefig(path + 'X1X2Save.png')

    plt.show()

def plot_error_c():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040n/"
    
    f = open(path + "X1-4to20.txt")
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
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040i/"
    
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
    
def take_avg_easy(value):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/"
    
    errorX1 = []
    errorX2 = []
    errorX1X2 = []
    errorC = []
    errorD = []
    
    #k
    #value = "4"
    
    #for iteration in range(1, 11):
        #pathn = path + str(iteration)
    for k in os.listdir(path):
        if k == "k" + str(value):
            pathnn = path + k
                        
            with open(pathnn + "/errsX1.txt") as X1:
                for line in X1:
                    errorX1.append(line)
                
            with open(pathnn + "/errsX2.txt") as X2:
                for line in X2:
                    errorX2.append(line)
                
            with open(pathnn + "/errsX1X2.txt") as X1X2:
                for line in X1X2:
                    errorX1X2.append(line)
            
            with open(pathnn + "/errsC.txt") as X1X2:
                for line in X1X2:
                    errorC.append(line)
            
            with open(pathnn + "/errsD.txt") as X1X2:
                for line in X1X2:
                    errorD.append(line)            
    
    fw1 = open(path + 'k' + str(value) + 'errX1.txt', "w")
    fw2 = open(path + 'k' + str(value) + 'errX2.txt', "w")
    fw12 = open(path + 'k' + str(value) + 'errX1X2.txt', "w")
    fwC = open(path + 'k' + str(value) + 'errC.txt', "w")
    fwD = open(path + 'k' + str(value) + 'errD.txt', "w")
    
    
    for i in errorX1:
        fw1.write(i + "\n")
        print i + "\n"
    
    for i in errorX2:
        fw2.write(i + "\n")
        print i + "\n"
    
    for i in errorX1X2:
        fw12.write(i + "\n")
        print i + "\n"
    
    for i in errorC:
        fwC.write(i + "\n")
        print i + "\n"
    
    for i in errorD:
        fwD.write(i + "\n")
        print i + "\n"
    
    
    fw1.close()
    fw2.close()
    fw12.close()
    fwC.close()
    fwD.close()

def take_avg_agg():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/60402/k"
    
    res = []
    
    for k in range(2,31):
        fname = path + str(k) + "errX1X2.txt"
        print fname
        with open(fname) as f:
            for line in f:
                res.append(line.strip())
                
    print res
    
    fw = open(path + 'AllerrX1X2.txt', "w")
    
    for i in range (2,21):
        for j in range(1,i):
            print str(i) + "," + str(j)
            fw.write (str(i) + "," + str(j) + "\t")
    
    fw.write("\n")
    
    for r in res:
        fw.write(r)
    
    fw.close()
    
    '''
    for k in os.listdir(path):
        if k == "k" + str(value):
            pathnn = path + k
                        
            with open(pathnn + "/errsX1.txt") as X1:
                for line in X1:
                    errorX1.append(line)
                
            with open(pathnn + "/errsX2.txt") as X2:
                for line in X2:
                    errorX2.append(line)
                
            with open(pathnn + "/errsX1X2.txt") as X1X2:
                for line in X1X2:
                    errorX1X2.append(line)
    
    
    fw1 = open(path + 'k' + str(value) + 'errX1.txt', "w")
    fw2 = open(path + 'k' + str(value) + 'errX2.txt', "w")
    fw12 = open(path + 'k' + str(value) + 'errX1X2.txt', "w")
    
    fwx = open(path + 'k' + str(value) + 'errX1X2Full.txt', "w")
    
    for i in errorX1:
        fw1.write(i + "\n")
        print i + "\n"
    
    for i in errorX2:
        fw2.write(i + "\n")
        print i + "\n"
    
    for i in errorX1X2:
        fw12.write(i + "\n")
        print i + "\n"
    fw1.close
    fw2.close
    fw12.close
    '''

def merge_errors(k):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/k" + str(k) + "/"
    errorsX1 = []
    errorsX2 = []
    errorsX1X2 = []
    errorsC = []
    errorsD = []
    for x in os.listdir(path):
        if x.startswith("c"):
            print x
            pathn = path + x
            with open(pathn + "/err.txt") as file:
                array2d = [line for line in file]
                errorsX1.append(array2d[1])
                errorsX2.append(array2d[3])
                errorsX1X2.append(array2d[4])
                errorsC.append(array2d[5])
                errorsD.append(array2d[6])
            
    fw1 = open(path + "/errsX1.txt", "w")
    for e in errorsX1:
        fw1.write(e.strip()+",")
    
    fw2 = open(path + "/errsX2.txt", "w")
    for e in errorsX2:
        fw2.write(e.strip()+",")
    
    fw12 = open(path + "/errsX1X2.txt", "w")
    for e in errorsX1X2:
        fw12.write(e.strip()+",")
    
    fwc = open(path + "/errsC.txt", "w")
    for e in errorsC:
        fwc.write(e.strip()+",")
    
    fwd = open(path + "/errsD.txt", "w")
    for e in errorsD:
        fwd.write(e.strip()+",")
    
    fw1.close()
    fw2.close()
    fw12.close()
    fwc.close()
    fwd.close()

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
    
def kmeans(clusters):
    
    pathp = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040fix/k22/c4d18/'
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040fix/k11/c4d7/'
    
    ptrn = []
    with open(pathp + 'pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())
    
    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/k20/c15d5/'

    #with open(path + 'vector.txt') as f:
    #    arrayV = [[float(digit) for digit in line.split('\t')] for line in f]
    with open(path + 'W1dW2dT.txt') as f:
        arrayV = [[float(digit) for digit in line.split('\t')] for line in f]
    
    V = np.array(arrayV)
    kmeans = KMeans(n_clusters = clusters, random_state = 0).fit(V)
    '''
    new_path = path + str(clusters)
    
    try:
        os.mkdir(new_path)
    except OSError:
        print ("Directory %s already exists!" %new_path)
    else:
        print ("Successfully created the directory %s " %new_path)
    '''
    pathc = path + "ClusterT/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
    
    fw = open(pathc + "Cluster" + str(clusters) +".txt", "w")
    fwc = open(pathc + str(clusters) +".txt", "w")
    fws = open(pathc + "Center" + str(clusters) + ".txt", "w")
    
    for i in range (len(kmeans.labels_)):
        print ptrn[i], kmeans.labels_[i]
        fw.write(ptrn[i] + "\t" + str(kmeans.labels_[i]) + "\n")
        fwc.write(str(kmeans.labels_[i]) + "\n")
    
    
    #fw_cent = open(paths + str(clusters) +".txt", "w")
        
    for i in range (len(kmeans.cluster_centers_)):
        center = ""
        print (kmeans.cluster_centers_[i])
        for j in range (len(kmeans.cluster_centers_[i])):
            center = center + "," + str(kmeans.cluster_centers_[i][j])
            #fws.write(str(kmeans.cluster_centers_[i][j]) + ",")
        fws.write(center[1:] + "\n")
        #fws.write("\n")
        #fws.write("%s\n" %kmeans.cluster_centers_[i])
        #fws.write(str(kmeans.cluster_centers_[i]) + "\n")
        
    
    '''
    fw_score = open(new_path + "/ClusteringScore" + str(clusters) +".txt", "w")
    
    labels = kmeans.labels_
    print(metrics.silhouette_score(V, labels, metric='euclidean'))
    print(metrics.calinski_harabaz_score(V, labels))
    print(davies_bouldin_score(V, labels))
    fw_score.write("Silhouette Score: " + str(metrics.silhouette_score(V, labels, metric='euclidean')) + "\n")
    fw_score.write("Calinski Harabaz Score: " + str(metrics.calinski_harabaz_score(V, labels)) + "\n")
    fw_score.write("Davies Bouldin Score: " + str(davies_bouldin_score(V, labels)))
    '''
    f.close()
    fw.close()
    fwc.close()
    fws.close()
    #fw_cent.close()
    #fw_score.close()

def spectral(n):
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/k11/c8d3/'
    
    with open(path + 'W1cW2cW1dW2d.csv') as f:
        array2d = list(csv.reader(f, quoting=csv.QUOTE_NONNUMERIC))
        
    X = np.array(array2d)
    
    '''
    txt file
    with open(path + "W1cW2cW1dW2d.txt") as f:
        array2d = [[float(digit) for digit in line.split('\t')] for line in f]

    X = np.array(array2d)
    '''

    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040k10/'
    ptrn = []
    with open('C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())

    clustering = SpectralClustering(n_clusters=n, assign_labels="discretize", random_state=0).fit(X)
    
    pathc = path + "Spectral/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "/Spectral" + str(n) + ".txt", "w")
    fwc = open(pathc + str(n) + ".txt", "w")
    
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    fwc.close
    f.close()
    
def hierarchical(n):
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040fix/k11/c4d7/'
    
    with open(path + "W1dW2d.txt") as f:
        array2d = [[float(digit) for digit in line.split('\t')] for line in f]

    X = np.array(array2d)

    #path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040k10/'
    ptrn = []
    with open(path + 'pattern.txt') as patterns:
        for p in patterns:
            ptrn.append(p.strip())

    clustering = AgglomerativeClustering(n_clusters=n, affinity='euclidean', linkage='ward').fit(X)
    
    print dir(clustering)
    
    pathc = path + "Hierarchical/"
    
    if (os.path.isdir(pathc) == False):
        os.mkdir(pathc)
        
    fw = open(pathc + "/Hierarchical" + str(n) + ".txt", "w")
    fwc = open(pathc + str(n) + ".txt", "w")
    
    for l in range(len(clustering.labels_)):
        fw.write(ptrn[l] + "\t" + str(clustering.labels_[l]) + "\n")
        fwc.write(str(clustering.labels_[l]) + "\n")

    fw.close()
    fwc.close
    f.close()

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
 
def to_plot():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/'
    fw = open(path + 'AllplotD.txt', 'w')
    s = ''
    for i in range(2,31):
        for j in range(1,i):
            s = s + str(i) + "," + str(j) + "\t"
    fw.write(s + '\n')
    s = ''
    for i in range(2,31):
        fname = path + 'k' + str(i) + 'errD.txt'
        with open(fname) as f:
            for line in f:
                s += line.strip()
    fw.write(s)
    fw.close()
    
def concateWcWd():
    path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040ab/k11/c8d3/'
    
    with open(path + 'W1c.csv') as fw1c:
        w1c = list(csv.reader(fw1c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W2c.csv') as fw2c:
        w2c = list(csv.reader(fw2c, quoting=csv.QUOTE_NONNUMERIC))
    
    with open(path + 'W1d.csv') as fw1d:
        w1d = list(csv.reader(fw1d, quoting=csv.QUOTE_NONNUMERIC))
        
    with open(path + 'W2d.csv') as fw2d:
        w2d = list(csv.reader(fw2d, quoting=csv.QUOTE_NONNUMERIC))
        
    w1cA = np.array(w1c)
    w2cA = np.array(w2c)
    w1dA = np.array(w1d)
    w2dA = np.array(w2d)
    
    w12cA = (w1cA + w2cA) /2
    
    w = np.concatenate((w12cA, w1dA, w2dA), axis = 1)
    
    np.savetxt(path + "W1cW2cW1dW2d.csv", w, delimiter=",")
    
    fw1c.close()
    fw2c.close()
    fw1d.close()
    fw2d.close()
    
    
if __name__ == "__main__":
    print ('Start:')
    #merge()
    for i in range(2,6):
    #    kmeans(i)
        spectral(i)
    #    hierarchical(i)
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
    #plot_error_c()
    #take_avg_agg()
    #plot_err()
    #-------------------
    #for i in range(2,31):
    #    merge_errors(i)
    #    take_avg_easy(i)
    #plot_error_no_errorbar()
    #to_plot()
    #merge_err()
    #plot_all()
    #concateWcWd()
    print ('End')