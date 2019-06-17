#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第1回演習問題
"""
import numpy as np

########## 課題1(a)
print("課題1(a)")
A = np.matrix([[1,2,3,4,5],
            [6,7,8,9,10],
            [11,12,13,14,15],
            [16,17,18,19,20]])
b = np.array([[1],[0],[1],[0],[1]])
print("A:\n{}\nb:\n{}".format(A,b))
########## 課題1(b)
print("課題1(b)")
Ab = np.dot(A,b)
print("行列Aとベクトルbの積:\n{}".format(Ab))
########## 課題1(c)
print("課題1(c)")
row_sum = A.sum(axis=1)
col_sum = A.sum(axis=0)
print("行列Aの行和: {}".format(row_sum))
print("行列Aの列和: {}".format(col_sum))
########## 課題1(d)-i.
print("課題1(d)-i")
a = 0
for i in range(10):
    a = 2*a+1
    print("a{} = {}".format(i+1,a))
    
########## 課題1(d)-ii.
print("課題1(d)-ii")
a1 = 6
for i in range(10):
    if a1%2 == 0:
        a1 = a1/2
        print("a{} = {}".format(i+1,a1))
    else:
        a1 = 3*a1+1
        print("a{} = {}".format(i+1,a1))