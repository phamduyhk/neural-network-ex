#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第7回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播および逆伝播の定義

##### データ数
n = 6000
n_test = 600

d = 28
m = 3     #クラス数

label = np.identity(m)

########## 確率的勾配降下法によるパラメータ推定
num_epoch = 100

e = []
e_test = []
error = []
error_test = []

##### 中間層のユニット数およびパラメータの初期値など


for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    for i in index:
        
        dat = np.genfromtxt("./train/{}.csv".format(i))[1:]
        
        xi = dat[1:].reshape((d, d)).T
        yi = label[int(dat[0])]
        
        ##### 順伝播
                
        ##### 誤差評価
        
        ##### 逆伝播
        
        ##### パラメータの更新

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    
    e = []
    
    ##### test error
    for j in range(0, n_test):
        dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
        
        xi = dat[1:].reshape((d, d)).T
        yi = label[int(dat[0])]
        
        ##### テスト誤差: 誤差をe_testにappendする

     ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする

     e_test = []
    

########## 誤差関数のプロット
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight')

########## confusion matrixの作成
ConfMat = np.zeros((m, m), int)

for j in range(0, n_test):        
    dat = np.genfromtxt("./test/{}.csv".format(j))[1:]
    
    xi = dat[1:].reshape((d, d)).T
    yi = label[int(dat[0])]
    
    ##### モデルの出力が最大のクラスにテストデータを分類する(ヒント: argmaxの引数にはモデルの出力を入れる)
    idx = np.argmax()
    ConfMat[int(dat[0]), idx] +=1
    
    ##### 正解ラベルと予測ラベルが異なるデータを出力
    if idx != int(dat[0]):
        
        plt.clf()
        sns.heatmap(xi, cbar =False, cmap="Blues", square=True)
        plt.axis("off")
        plt.title('{} to {}'.format(int(dat[0]), idx))
        plt.savefig("./misslabeled{}.pdf".format(j), bbox_inches='tight', transparent = True)

plt.clf()
sns.heatmap(ConfMat, annot = True, fmt="1", cbar =False, cmap="Blues")
plt.savefig("./confusion.pdf", bbox_inches='tight')
