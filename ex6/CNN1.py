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
    return x*(x>0)

def Convolution(img, param, bias, fncs, channel, padding, stride):
    ##### 課題1-(a). 畳み込み層の計算を完成させる
    pad = np.pad(img, [(0, 0), (0, 0), (padding, padding),
                       (padding, padding)], 'constant')
    H,W = img.shape
    filter_h = param.shape[0]
    filter_w = param.shape[1]
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    conv = np.zeros((1, channel, filter_h, filter_w, out_h, out_w))
    
    return 

def MaxPooling(img, filter_size, channel, stride):
    ##### 課題1-(b). maxプーリングの計算を完成させる
    
    return 


##### データの読み込み
img = np.load("./img.npy")
d = img.shape[0]

# ##### 課題2-(a)
# w = 

# ##### 課題2-(b)
# ##### 畳み込み
# C = 

# ##### プーリング
# P =    

# ########## 結果の出力
# plt.clf()
# sns.heatmap(img, cbar =False, cmap="CMRmap", square=True)
# plt.axis("off")
# plt.savefig("./original.pdf", bbox_inches='tight', transparent = True)

# for i in range(C.shape[2]):
#     plt.clf()
#     sns.heatmap(C[:,:,i], cbar =False, cmap="CMRmap", square=True)
#     plt.axis("off")
#     plt.title('Convolution1: axis{}'.format(i))
#     plt.savefig("./conv1-{}.pdf".format(i), bbox_inches='tight', transparent = True)

# for i in range(P.shape[2]):
#     plt.clf()
#     sns.heatmap(P[:,:,i], cbar =False, cmap="CMRmap", square=True)
#     plt.axis("off")
#     plt.title('Pooling: axis{}'.format(i))
#     plt.savefig("./pooling{}.pdf".format(i), bbox_inches='tight', transparent = True)    





