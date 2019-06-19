import numpy as np
def exp(x):
    return np.exp(x)
def softmax(x):
    ##### 課題1(b)
    # ソフトマックス関数を返すプログラムを書く
    # 1/(1+np.exp(-x))
    r = np.apply_along_axis(lambda a: exp(a), 0, x)
    return r

x = np.array([1,4])
print(x.shape)
print(softmax(x))
