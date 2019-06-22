import numpy as np
def exp(x):
    return np.exp(x)
def softmax(x):
    ##### 課題1(b)
    # ソフトマックス関数を返すプログラムを書く
    # 1/(1+np.exp(-x))
    r = np.apply_along_axis(lambda a: exp(a), 0, x)
    return r


def ReLU(x):
    # 課題1(a)
    # ReLUとその微分を, この順番で返すプログラムを書く
    v = np.maximum(0,x)
    grad = np.zeros(x)
    grad[x>0] = 1
    return v,grad


def backward(w, delta, derivative):
    # 課題1(e)
    # 逆伝播のプログラムを書く
    return np.dot(w.T, delta*derivative)


x = np.array([0,1,2,3,4])

print(x.shape)
print(softmax(x))

a,da = [], []
for item in x:
    v,dv = ReLU(item)
    a.append(v)
    da.append(dv)
print(a)
print(da)


w = np.random.normal(0, 0.3, size=(2,))

w2 = np.random.normal(0, 0.3, size=(5,))

print(w.reshape(2,1))
print(np.dot(w.reshape(2,1),w2.reshape(1,5)).shape)

delta = np.random.normal(0, 0.3, size=(2,5))
derivative = np.random.normal(0, 0.3, size=(2, 5))

backward = backward(w,delta,derivative)
print(backward.shape)
# print(backward)

b = delta[:,1:]
print(delta)
print(b)
