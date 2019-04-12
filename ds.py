# -*- coding: utf-8 -*-
"""
Created on Sun Mar 03 19:05:49 2019

@author: mirza
"""

import math as m

sum = 0.0

for i in range(519):
    j = i + 1
    k = 1.0/(j*(j+2))
    #print k
    sum += k

#print sum
    
print 3 * (m.pow(3,18)-1) + 6 * (m.pow(2,18) - 1) + 5
print (m.pow(3,19)-1) + 3 * (m.pow(2,19) - 1)

print m.pow(3,19) + 157430

print 1.0/3 + 1.0/8 + 1.0/15 + 1.0/24 + 1.0/35 

print m.pow(2,8)

print 0.5 * (1.5 - 1.0/570 - 1.0/571)