#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第3回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播


##### データ数
n = 10000
n_test = 1000

d = 28**2
m = 4     #クラス数

label = np.identity(m)

########## 確率的勾配降下法によるパラメータ推定
num_epoch = 10

eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []

##### 中間層のユニット数とパラメータの初期値


##### 誤差逆伝播法によるパラメータ推定

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch+1) 
    for i in index:
        dat = np.genfromtxt("./train/{}.csv".format(i))[1:]
        
        xi = np.append(1, dat[1:])
        yi = label[int(dat[0])]

        ##### 順伝播
        
        ##### 誤差評価: 誤差をeにappendする
        
        ##### 逆伝播

        ##### パラメータの更新

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする

    e = []
    
    ##### test error
    for j in range(0, n_test):        
        dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
        
        xi = np.append(1, dat[1:])
        yi = label[int(dat[0])]
        
        ##### テスト誤差: 誤差をe_testにappendする

    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする

    e_test = []
    

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf")


########## confusion matrixの作成
ConfMat = np.zeros((m, m), int)

for j in range(0, n_test):        
    dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
    
    xi = np.append(1, dat[1:])
    yi = label[int(dat[0])]
        
    ##### モデルの出力が最大のクラスidxにテストデータを分類する
    idx = 
    ConfMat[int(dat[0]), idx] +=1

    ##### 正解ラベルと予測ラベルが異なるデータを出力
    if idx != int(dat[0]):
        D = np.reshape(dat[1:], (28, 28)).T
        
        plt.clf()
        sns.heatmap(D, cbar =False, cmap="Blues", square=True)
        plt.axis("off")
        plt.title('{} to {}'.format(int(dat[0]), idx))
        plt.savefig("./misslabeled{}.pdf".format(j), bbox_inches='tight')

plt.clf()
sns.heatmap(ConfMat, annot = True, fmt="1", cbar =False, cmap="Blues")
plt.savefig("./confusion.pdf")

