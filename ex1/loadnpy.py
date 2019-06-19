import numpy as np
import sys
args = sys.argv
def load(filename):
    return np.load(filename)


if __name__ == '__main__':
    npy = load(args[1])
    print(npy)
    