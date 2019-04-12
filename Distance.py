# -*- coding: utf-8 -*-
"""
Created on Mon Apr 08 19:10:17 2019

@author: mirza
"""

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

#print levenshteinDistance('Fss','Fss')

path = 'C:/Project/EDU/files/2013/example/Topic/60/LG/'

pat = []
with open(path + 'pattern.txt') as file:
    for line in file:
        pat.append(line.strip())
        
#print pat

for p in pat:
    print p,
print
for p1 in pat:
    print p1,
    for p2 in pat:
        print levenshteinDistance(p1,p2),
    print
