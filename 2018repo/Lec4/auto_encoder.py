#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第4回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

##### 
def sigmoid(x):
    tmp = 1/(1+np.exp(-x))
    return tmp, tmp*(1-tmp)

def ErrorFunction(x, y):
    # returnの後に二乗誤差関数を返すプログラムを書く
    return

def forward(x, param, fncs):
    tmp = np.dot(param, x)
    z = np.append(1, fncs(tmp)[0])
    u = fncs(tmp)[1]
    return z, u

def backward(param, delta, derivative):
    return np.dot(param.T, delta)*derivative

def adam(param, m, v, error, t, 
         alpha = 0.001, beta1 = 0.9, beta2 = 0.999, tol = 10**(-8)):
    # returnの後にadamによるパラメータ更新のプログラムを書く
    return 



##### データ数
n = 10000
n_test = 1000

d = 28**2

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 10

#####中間層のユニット数
N1 = 64

w0 = np.random.normal(0, 0.3, size=(N1, d+1))
w1 = np.random.normal(0, 0.3, size=(d, N1+1))

##### adamのパラメータの初期値
m0 = np.zeros(shape=(N1, d+1))
v0 = np.zeros(shape=(N1, d+1))
m1 = np.zeros(shape=(d, N1+1))
v1 = np.zeros(shape=(d, N1+1))


for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    for i in index:
        dat = np.genfromtxt("./train/{}.csv".format(i))[1:]
        
        ##### データに加えるノイズ
        noise = 
        xi = np.append(1, dat[1:] + noise)

        ##### 順伝播
        z1, u1 = forward(xi, w0, sigmoid)
        z2 = np.dot(w1, z1)
               
        ##### 誤差評価
        e.append(ErrorFunction(z2, dat[1:]))
        
        ##### 逆伝播
        delta2 = z2 - dat[1:]
        delta1 = backward(w1[:, 1:], delta2, u1)
        
        ##### adamによるパラメータの更新

    
    ##### training error
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):        
        dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
        
        ##### データに加えるノイズ
        noise = 
        xi = np.append(1, dat[1:] + noise)

        z1, u1 = forward(xi, w0, sigmoid)
        z2 = np.dot(w1, z1)
        
        e_test.append(ErrorFunction(z2, dat[1:]))
    
    error_test.append(sum(e_test)/n_test)
    e_test = []


########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf")

##### 0から999までの自然数のうち, 好きな数をjに代入する
j = 

dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
xi = np.append(1, dat[1:] + np.random.normal(0, 0.05, size= d))
z1, u1 = forward(xi, w0, sigmoid)
z2 = np.dot(w1, z1)        

D_o = np.reshape(dat[1:], (28, 28)).T
D = np.reshape(z2, (28, 28)).T


plt.clf()
sns.heatmap(D_o, cbar =False, cmap="Blues", square=True)
plt.axis("off")
plt.savefig("./original.pdf", bbox_inches='tight')

plt.clf()
sns.heatmap(D, cbar =False, cmap="Blues", square=True)
plt.axis("off")
plt.savefig("./reconstruct.pdf", bbox_inches='tight')

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=8,figsize=(10, 10))
for i in range(8):
    for j in range(8):
        D = np.reshape(w0[i, 1:], (28, 28)).T
        axes[i,j].imshow(D)
        axes[i,j].axis("off")
plt.savefig("./layer1.pdf", bbox_inches='tight')

plt.clf()
fig,axes = plt.subplots(nrows=8,ncols=8,figsize=(10, 10))
for i in range(8):
    for j in range(8):
        D = np.reshape(w1[:, i], (28, 28)).T
        axes[i,j].imshow(D)
        axes[i,j].axis("off")
plt.savefig("./layer2.pdf", bbox_inches='tight')

