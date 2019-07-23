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
    return x*(x > 0)


def Convolution(img, param, bias, fncs, channel, padding, stride):
    """
    @param
    img.shape = (channel, h,w)
    param.shape = (channel,h,w)
    bias.shape = (channel)
    fncs: ReLU, sigmoid, softmax ...

    @return
    convolution matrix with shape = (channel,height,weight)
    """
    ##### 課題1-(a). 畳み込み層の計算を完成させる
    image_padded = np.pad(img, [(0, 0), (padding, padding),
                                (padding, padding)], 'constant')
    H, W = img.shape[1], img.shape[2]
    filter_h = param.shape[1]
    filter_w = param.shape[2]
    out_h = (H + 2*padding - filter_h)//stride + 1
    out_w = (W + 2*padding - filter_w)//stride + 1
    conv = np.zeros((channel, out_h, out_w))
    for c in range(channel):
        for i in range(out_h):
            for j in range(out_w):
                for p in range(filter_h):
                    for q in range(filter_w):
                        conv[c, i, j] += param[c, p, q] * image_padded[c,
                                                                       stride * (i-1)+p+1, stride*(j-1)+q+1]+bias[c]
    conv = fncs(conv)
    return conv


def MaxPooling(img, filter_size, channel, stride):
    """
    @param
    img.shape = (channel, h,w)
    filter.height = filter.weight = filter_size

    @return
    convolution matrix with shape = (channel,height,weight)
    """
    ##### 課題1-(b). maxプーリングの計算を完成させる
    H, W = img.shape[1], img.shape[2]
    out_h = (H - filter_size)//stride + 1
    out_w = (W - filter_size)//stride + 1
    P = np.zeros((channel, out_h, out_w))
    for c in range(channel):
        for i in range(out_h):
            for j in range(out_w):
                inside_filter = []
                for p in range(filter_size):
                    for q in range(filter_size):
                        inside_filter.append(
                            img[c, stride*(i-1)+p, stride*(j-1)+q])
                P[c, i, j] = max(inside_filter)
    return P


# データの読み込み
img = np.load("./img.npy")
d = img.shape[0]

# 課題2-(a)
W1 = np.array([[-1, -1, 0],
               [-1, 0, 1],
               [0, 1, 1]])
W2 = np.array([[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]])
W3 = np.ones((3, 3))
w = np.array([W1, W2, W3])
print(w)
# 課題2-(b)
# 畳み込み
img_with_3channel = np.array([img, img, img])
print("Img shape {}".format(img_with_3channel.shape))
print(img_with_3channel)
bias = np.zeros((3))
C = Convolution(img_with_3channel, w, bias, ReLU, 3, 2, 1)
print("Convolution shape {}".format(C.shape))
print(C)

# プーリング
P = MaxPooling(C, 3, 3, 3)
print("Pooling shape {}".format(P.shape))
print(P)

# 結果の出力
plt.clf()
sns.heatmap(img, cbar=False, cmap="CMRmap", square=True)
plt.axis("off")
plt.savefig("./original.pdf", bbox_inches='tight', transparent=True)

for i in range(C.shape[0]):
    plt.clf()
    sns.heatmap(C[i, :, :], cbar=False, cmap="CMRmap", square=True)
    plt.axis("off")
    plt.title('Convolution1: axis{}'.format(i))
    plt.savefig("./conv1-{}.pdf".format(i),
                bbox_inches='tight', transparent=True)

for i in range(P.shape[0]):
    plt.clf()
    sns.heatmap(P[i, :, :], cbar=False, cmap="CMRmap", square=True)
    plt.axis("off")
    plt.title('Pooling: axis{}'.format(i))
    plt.savefig("./pooling{}.pdf".format(i),
                bbox_inches='tight', transparent=True)
