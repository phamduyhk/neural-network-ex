import numpy as np
def t(x):
    x = x + 1
    return x + 5

x = 5
t(x)
print("{},{}".format(x,t(x)))

a = np.array([1,2,3,4,5])
print(a[1:5])
for i in range(5)[::-1]:
    print(i)

Z = np.zeros(shape=(7,3))
yi = np.array([3,1,3])
print(Z-yi)
print((Z-yi).shape)