# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 20:07:31 2019

@author: mirza
"""

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

with open(path + 'W2d.txt') as file:
    array2d = [[float(digit) for digit in line.split('\t')] for line in file]
    
labels = []
for array in array2d:
    #print array
    c = 0
    cluster = 0
    l = len(array)
    #print (l)
    max = array[0]
    
    for i in range(l):
        #print i
        if (array[i] > max):
            #print array[i]
            cluster = c
            max = array[i]
        c = c + 1
    #print
    print cluster
    labels.append(cluster)

fw = open(path + 'W2dCluster.txt', "w")

for l in labels:
    print l
    fw.write(str(l)+"\n")

fw.close()