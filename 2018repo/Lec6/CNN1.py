#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第6回演習問題
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)

#####
def ReLU(x):
    # returnの後にシグモイド関数を返すプログラムを書く
    return x*(x>0)

def Convolution(img, param, bias, fncs, size, channel, padding, stride):
    ##### 課題1-(a). 畳み込み層の計算を完成させる

    return 

def MaxPooling(img, size, channel, stride):
    ##### 課題1-(b). maxプーリングの計算を完成させる
    
    return 


##### データの読み込み
img = np.load("./img.npy")
d = img.shape[0]

##### 課題2-(a)
w = 

##### 課題2-(b)
##### 1回目の畳み込み
C1 = 

##### プーリング
P =    

##### 2回目の畳み込み
C2 = 

########## 結果の出力
plt.clf()
sns.heatmap(img, cbar =False, cmap="mako_r", square=True)
plt.axis("off")
plt.savefig("./original.pdf", bbox_inches='tight', transparent = True)

for i in range(size1):
    plt.clf()
    sns.heatmap(C1[:,:,i], cbar =False, cmap="mako_r", square=True)
    plt.axis("off")
    plt.title('Convolution1: axis{}'.format(i))
    plt.savefig("./conv1-{}.pdf".format(i), bbox_inches='tight', transparent = True)

for i in range(size1):
    plt.clf()
    sns.heatmap(P[:,:,i], cbar =False, cmap="mako_r", square=True)
    plt.axis("off")
    plt.title('Pooling: axis{}'.format(i))
    plt.savefig("./pooling{}.pdf".format(i), bbox_inches='tight', transparent = True)    

for i in range(size2):
    plt.clf()
    sns.heatmap(C2[:,:,i], cbar =False, cmap="mako_r", square=True)
    plt.axis("off")
    plt.title('Convolution2: axis{}'.format(i))
    plt.savefig("./conv2-{}.pdf".format(i), bbox_inches='tight', transparent = True)






