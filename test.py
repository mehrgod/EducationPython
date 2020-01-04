# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 18:07:24 2019

@author: mirza
"""

import numpy as np
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.linalg import orth
plt.style.use('seaborn-whitegrid')
#import plotly.plotly as py
#import plotly.graph_objs as go
from sklearn.cross_decomposition import CCA
#from scipy import stats
import os
import seaborn as sns
#from scipy import stats
import random as rnd

def test_subtract_array():
    A = np.array([[ 1,   2,   3,   4,   5 ],
                  [ 6,   7,   8,   9,   10],
                  [ 11,  12,  13,  14,  15],
                  [ 16,  17,  18,  19,  20],
                  [ 21,  22,  23,  24,  25]])
    
    B = 25 - A
    
    print B
    print np.amax(B)

def test_multi_plot():
    a = [1, 2, 3, 4, 5]
    b = [10, 11, 12, 13, 14]
    c = [0.1, 0.2 , 0.1, 0.2, 0.1]
    
    plt.errorbar(a, b, c, capsize = 5, label='test', capthick=1, linewidth=2, elinewidth=1, linestyle = 'dashed');
    
    plt.errorbar(b, a, yerr=c, capsize = 5, label='test', capthick=1, linewidth=2, elinewidth=1, linestyle = 'dotted');
    
    plt.show()
    

def test_sort():
    a = []
    
    a.append(2)
    a.append(4)
    a.append(3)
    
    print a
    
    a.sort(reverse = True)
    
    print a
    
    a.sort()
    
    print a

def test_jump_array():
    
    for i in range(1):
        print i

def test_minmax():
    A = np.array([[ 1,   2,   3,   4,   5 ],
                  [ 6,   7,   8,   9,   10],
                  [ 11,  12,  13,  14,  15],
                  [ 16,  17,  18,  19,  20],
                  [ 21,  22,  23,  24,  25]])
    
    print np.amax(A)
    print np.amin(A)

def test_number():
    a = [1, 2, 3]
    print a[1]*2
    
    for i in np.arange(0,1,0.1):
        print i

def test_epoch():
    
    n = 5
    
    
    eG = [[] for i in range(n)]
    
    print eG
    '''
    for i in range(n):
        eG.append(i)
    
    print eG
    
    eG[0].append('0')
    eG[2].append('2')
    eG[1].append('1')
    eG[2].append('22')
    
    
    print eG
    '''
    for e in range(10):
        
        #learning_rate = 0.1/np.sqrt(e+1)
        #index.append(e)
        
        for i in range(n):
            eG[i].append(e)
            '''
            print 'e %s' %e
            print 'i %s' %i
            print eG[0]
            print len(eG[i])
            print '====='*4
            '''
            #print e, i
            #print len(eG[i])
    print eG[0]
    print eG[1]
            
def test_normal_matrix():
    lower = 0
    upper = 0.2
    
    A = np.array([[ 1,   2,   3,   4,   5 ],
                  [ 6,   7,   8,   9,   10],
                  [ 11,  12,  13,  14,  15],
                  [ 16,  17,  18,  19,  20],
                  [ 21,  22,  23,  24,  25]])
    
    #A = np.array(arr)
    
    m, n = A.shape
    
    print m, n
    
    l = np.ndarray.flatten(A)
    
    minl = float(np.amin(l))
    maxl = float(np.amax(l))
    
    print minl
    print maxl
    
    print l
    
    l_norm = [ upper * (x - minl) / (maxl - minl) for x in l]
    
    print l_norm
    
    nA = np.reshape(l_norm, (m, n))
    
    print nA

def test_list():
    a = []
    
    b = []
    c = []
    
    b.append('1')
    b.append('2')
    c.append('3')
    c.append('4')
    
    a.append(b)
    #b.append('5')
    a.append(c)
    
    a[0].append('5')
    a[1].append('6')
    a[1].append('7')
    
    print b
    print c
    print a

def test_array():
    A = np.array([1, 2, 3])
    print A[0]
    
    B = []
    
    B[0] = 1
    
    print B

def test_orth():
    m = 4
    k = 3
    G_temp = np.random.normal(0,1,[m,k])

    G = orth(G_temp)
    
    print G_temp.shape
    print G.shape

def test_transpose():
    A = np.array([[ 1,   2,   3,   4,   5 ],
                  [ 6,   7,   8,   9,   10],
                  [ 11,  12,  13,  14,  15],
                  [ 16,  17,  18,  19,  20],
                  [ 21,  22,  23,  24,  25]])
    
    print np.transpose(A)
    print A.T

def test_random_select():
    n = 2
    
    A = np.array([[ 1,   2,   3,   4,   5 ],
                  [ 6,   7,   8,   9,   10],
                  [ 11,  12,  13,  14,  15],
                  [ 16,  17,  18,  19,  20],
                  [ 21,  22,  23,  24,  25]])
    
    test = []
    index = [i for i in range(A.shape[1])]
    
    for i in range(0, A.shape[0]):
        arr = rnd.sample(index, n)
        print arr
        for k in arr:
            A[i, k] = 0
            test.append((i, k))
            
    print A
    print test
    
    
def test_find_average():
    path = 'C:/Project/EDU/files/2013/example/Topic/60'
    all = []
    with open(path + "/seq.txt") as file:
        for line in file:
            l = line.split()
            all.append(len(l))
    
    s = 0
    for a in all:
        s = s + a
        
    print s
    print len(all)
    print float(s/len(all))

def test_replace():
    A = np.array([[ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1.]])
    
    B = np.array([[ 0.1,  0.2],
       [ 0.3,  0.4]])
    
    A[2:4, 3:] = B
    
    print A

def test_eigen():
    A = np.array([
        [1, -3, 3],
        [3, -5, 3],
        [6, -6, 4],])
    
    B = np.array([
        [0, 1],
        [-2, -3],])
    
    v, w = np.linalg.eig(A)
    
    k = 2
    
    [eigVal, G_new] = np.linalg.eig(A)
    print eigVal
    print G_new
    idx = eigVal.argsort()[::-1]
    eigVal = eigVal[idx]
    G_new = G_new[:, idx]
    G_new = G_new[:, 0:k]
    G_new = np.real(G_new)
    
    
    
    print G_new

def test_multicolumn():
    '''
    A = np.array([
        [1, 2, 3],
        [-1, 1, 2],
        [-1, 0, -1],])
    
    B = np.array([
        [1, 1, 0],
        [0, 2, 0],
        [-1, 3, 1],])
    
    
    C = np.array([
        [-1, 2, 1],
        [1, -1, 2],
        [-1, 0, 3],])
    '''
    A = np.array([
        [1, 4],
        [3, 2],])
    
    B = np.array([
        [-1, 0],
        [1, 1],])
    
    C = np.array([
        [0, 2],
        [1, -1],])
    
    
    
    print np.dot(A,B)
    print np.dot(B,C)
    
    print np.dot(np.dot(A, B), C)
    print np.dot(A, np.dot(B,C))


def column_center(X, m):
    avg = np.mean(X, axis=0)
    
    print 'original'
    print X
    
    new_row = []
    
    for row in X:
        new_row.append(row - avg)
        
    return np.vstack(new_row)
    
    #return np.mean(X, axis=0).reshape((1, m))

def test_matrix_init():
    
    Wx = np.random.rand(10, 10)
    Wy = np.random.rand(10, 10)
    
    print Wx
    print ('*'*10)
    print Wy

def test_zero_mean():
    A = np.array([
        [1, 1, 0, 2],
        [0, 2, 0, 3],
        [-1, 3, 1, 4],])
    
    #B = stats.zscore(A)
    
    B = column_center(A,3)
    print 'column_centered'
    print B

def first_1000():
    c = 0
    
    path = "C:/Project/Dataset/kdd10/algebra_2005_2006"
    
    fw = open(path + "/algebra_2005_2006_train_1000.txt", "w")
    
    with open(path + "/algebra_2005_2006_train.txt") as file1:
        for line in file1:
            c = c + 1
            if c < 1000:
                fw.write(line)
                
    fw.close()
    file1.close()


def test_sub_matrix():
    A = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        ])
    
    #A[2:, 2:] = 0
    
    print A[2:, :1]
        
    print A
    

def test_mean_squared():
    A = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [6, 0, 0],])
    
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],])
    
    print np.sqrt(np.mean((A - B) ** 2))

def test_subplot():
    
    x = range(10)
    y = range(10)
    
    fig, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
    '''
    plt.plot(x,y)
    plt.plot(y,x)
    '''
    
    ax1.plot(x,y)
    ax1.legend("1")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("y1")
    
    ax2.plot(x,y)
    ax2.legend("2")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("y1")
    
    #plt.legend([ax1, ax2],["1", "2"])
    '''
    for row in ax:
        for col in row:
            col.plot(x, y)
    '''        
    plt.show()
    

def error_bar_test():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/1/k15/c10d5/"
    with open(path + 'mean.csv') as file:
        array_mean = [[float(digit) for digit in line.split(',')] for line in file]
    
    with open(path + 'ci.csv') as file:
        array_ci = [[float(digit) for digit in line.split(',')] for line in file]
    
   
    x = ['W1c1','W1c2','W1c3','W1c4','W1c5','W1c6','W1c7','W1c8','W1c9','W1c10','W1d1','W1d2','W1d3','W1d4','W1d5','W2d1','W2d2','W2d3','W2d4','W2d5']
    #x = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
    
    sns.set()
    
    plt.figure(figsize = (20,10))
    
    y5 = array_mean[0]
    dy5 = array_ci[0]
    
    #plt.errorbar(x, y5, yerr=dy5, fmt='o', capsize = 5, label='C5', uplims=True);
    #capsize=5,capthick=2,ms=9,markerfacecolor='none'
    plt.errorbar(x, y5, yerr=dy5, capsize = 5, label='C5', capthick=1, elinewidth=1);
    #b[0].set_marker('_')
    #b[0].set_markersize(20)
    
    y4 = array_mean[1]
    dy4 = array_ci[1]
    
    plt.errorbar(x, y4, yerr=dy4, capsize = 5, label='C4', capthick=1, elinewidth=1);
    
    y3 = array_mean[2]
    dy3 = array_ci[2]
    
    plt.errorbar(x, y3, yerr=dy3, capsize = 5, label='C3', capthick=1, elinewidth=1);
    
    y2 = array_mean[3]
    dy2 = array_ci[3]
    
    plt.errorbar(x, y2, yerr=dy2, capsize = 5, label='C2', capthick=1, elinewidth=1);
    y1 = array_mean[4]
    dy1 = array_ci[4]
    
    plt.errorbar(x, y1, yerr=dy1, capsize = 5, label='C1', capthick=1, elinewidth=1);
    
    y0 = array_mean[5]
    dy0 = array_ci[5]
    
    plt.errorbar(x, y0, yerr=dy0, capsize = 5, label='C0', capthick=1, elinewidth=1);
    
    
    plt.legend()
    
    plt.savefig(path + 'cluster6.pdf')
    
    plt.show()
    
    
    '''
    x = np.linspace(0, 10, 50)
    dy = 0.8
    y = np.sin(x) + dy * np.random.randn(50)

    (_, caps, _) = plt.errorbar(x, y, yerr=dy, fmt='o', capsize = 5);
    for cap in caps:
        cap.set_markeredgewidth(1)
    '''


def test_heatmap():
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10-2/1/k15/c10d5/"
    with open(path + 'heatmap.csv') as file:
        array2d = [[float(digit) for digit in line.split(',')] for line in file]
        
    X = np.array(array2d)
    print X.shape
    
    sns.set()
    plt.figure(figsize = (10,15))
    #plt.imshow(X, cmap='hot', interpolation='nearest')
    y_axix_labels = ['SS_','Ss_','_Ss','_FF','_Ss_','_FS','_SS','FS_','FFS','_Fs','_Fs_','Fs_','_FS_','_FFS','_Ff','ee_','FfS','_FSs','fS_','FSs','Sss','ss_','FSs_','_ss','_ss_','sS_','fSs','FFf','Sssss','Ssss','FFs','_FFs','_FFf','ssss_','Sss_','_Ffs_','_Sss','Ffs_','FSss','sss_','_ee','fss','fss_','eee_','Fff','_Ssss','sss','sfs','_ee_','_Sss_','Ssss_','_Ffs','ssss','_eee_','Fsss','_Fsss','fff','Ffs','ffs','ffs_','eeee','fssss','fsss','eee','_eee','_Fss','_Fss_','Fss','Fss_','_fs_','sssss_','_fs','_ff','_Fff','sssss','fs_']
    x_axix_labels = ['W1c','W1c','W1c','W1c','W1c','W1c','W1c','W1c','W1c','W1c','W1d','W1d','W1d','W1d','W1d','W2d','W2d','W2d','W2d','W2d']
    sns_plot = sns.heatmap(X, yticklabels=y_axix_labels, xticklabels=x_axix_labels)
    plt.show()
    sns_plot.figure.savefig(path + "/heatmap.pdf")
    #plt.savefig()

def test_row_matrix():
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    #print X
    
    Y = np.array([
        [5, 3, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ])
    print Y[0]
'''
def test_take_avg_easy(value):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10/10/"
    
    for k in os.listdir(path):
        if k == "k" + str(value):
            pathnn = path + k
            print pathnn

'''
def test_loop(k):
    path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040i10/1/k" + str(k) + "/"
    #path = "C:/Project/EDU/files/2013/example/Topic/60/LG/6040_184/1/k" + str(k) + "/"
    errorsX1 = []
    errorsX2 = []
    errorsX1X2 = []
    errorsC = []
    errorsD = []
    for i in range(1,k):
        x = 'c' + str(i) + 'd' + str(k-i)
        print path + x
    
    for x in os.listdir(path):
        if x.startswith("c"):
            #print x
            pathn = path + x
            print pathn
            with open(pathn + "/err.txt") as file:
                array2d = [line for line in file]
                errorsX1.append(array2d[1])
                errorsX2.append(array2d[3])
                errorsX1X2.append(array2d[4])
                errorsC.append(array2d[5])
                errorsD.append(array2d[6])
    


def test_colum_center(X, m):
    #return X - np.mean(X, axis=0).reshape((1, m))
    return np.mean(X, axis=0).reshape((1, m))

def test_avg_std():
    a = [1,2,3]
    print np.average(a)

def test_list_to_str(l):
    s = ''
    for i in l:
        s = s + ',' + str(i)
    return s[1:]

def test_take_avg_vec(l):
    vecs = []
    for line in l:
        vec = [float(digit) for digit in line.split(',')]
        #print vec
        vecs.append(vec)
        
    print np.average(vecs, axis=0)
    print np.std(vecs, axis=0)
    return np.average(vecs, axis=0)

def test_substr():
    s = '123456'
    
    print s[:1]
    print s[1:-1]

def test_cca():
    X = [[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]]
    Y = [[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]]
    
    cca = CCA(n_components = 2)
    cca.fit(X, Y)
    
    CCA(copy=True, max_iter=500, n_components=2, scale=True, tol=1e-06)
    
    X_c, Y_c = cca.transform(X, Y)
    
    #cca.fit_transform(X, xcca)
    
    #print xcca
    
    print X_c
    print Y_c
    

def test_seaborn():
    #sns.set(style="darkgrid")
    #X = [1,2,3]
    #Y = [4,5,6]

    #df = sns.load_dataset("anscombe")

# Show the results of a linear regression within each dataset
    '''
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
    '''
'''
def test_plotly():
    
    X = [1,2,3]
    Y = [4,5,6]
    
    trace1 = go.Scatter(X=[1,2],Y=[1,2])
    trace2 = go.Scatter(X=[1,2],Y=[2,1])
    py.iplot([trace1, trace2])
    
    N = 500
    
    random_x = np.linspace(0, 1, N)
    random_y = np.random.randn(N)
    
    # Create a trace
    trace = go.Scatter(
        x = random_x,
        y = random_y
    )
    
    data = [trace]
    
    py.iplot(data, filename='basic-line')

    
    random_x = np.linspace(0,1,10)
    print random_x
    random_y = np.random.randn(10)
    print random_y
    
    trace = go.Scatter(X,Y)
    data = [trace]
    py.iplot(data)
    
'''
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
    
    
def test_islower():
    s = "AaBb"
    print s.lower()

def test_sub():
    s = ",122334535gsf"
    print s[1:]

if __name__ == "__main__":
    print('Start')
    #test_row()
    #test_row_matrix()
    #test_heatmap()
    #error_bar_test()
    #test_subplot()
    #test_mean_squared()
    #test_sub_matrix()
    #first_1000()
    #test_zero_mean()
    #test_matrix_init()
    #test_multicolumn()
    #test_eigen()
    #test_replace()
    #test_find_average()
    #test_random_select()
    #test_transpose()
    #test_islower()
    #test_sub()
    #test_orth()
    #test_array()
    #test_list()
    #test_epoch()
    #test_normal_matrix()
    #test_number()
    #test_minmax()
    #test_jump_array()
    #test_sort()
    #test_multi_plot()
    test_subtract_array()
    
    print ('END')