import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def get_dataset(function, num):
    xmin = -1
    xmax = 1
    np.random.seed(1234)
    X = np.random.rand(num, 2)*(xmax-xmin)+xmin
    x1 = X[:, 0]
    x2 = X[:, 1]
    if function == 'sphere':
        y = (x1 ** 2 + x2 ** 2) / 2
    elif function == 'sample':
        y = (1+np.sin(4*math.pi*x1))*x2/2
    else:
        y = (x1+x2) / 2
    y = y[:, np.newaxis]
    return X, y


if __name__ == '__main__':
    X, y = get_dataset('sample', 10000)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(X[:, 0], X[:, 1], y)
    plt.show()
