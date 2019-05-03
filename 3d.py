# -*- coding: utf-8 -*-
"""
Created on Wed May 01 22:32:22 2019

@author: mirza
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[10,10,10,10,10,10,10]
y =[2 ,3 ,4 ,5 ,6 ,7 ,8]
z =[0.012955659,0.013028414,0.013132203,0.013206596,0.013286289,0.013335345,0.013305571]

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/6040b/'

with open(path + 'errX1X2.txt') as file:
    err = [[float(digit) for digit in line.split('\t')] for line in file]

with open(path + 'k.txt') as file:
    k = [[int(digit) for digit in line.split('\t')] for line in file]

with open(path + 'kc.txt') as file:
    kc = [[int(digit) for digit in line.split('\t')] for line in file]

print len(err)
print len(k)
print len(kc)

ax.scatter(k, kc, err, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()