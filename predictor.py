from dataset_generator import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from argparse import ArgumentParser


def predict(test_function, number, num_layers):
    dataset_X, dataset_y = get_dataset(test_function, number)
    train_X, test_X, train_y, test_y = train_test_split(dataset_X, dataset_y, test_size=0.2, random_state=1234)
    # Layer
    class Layer:
        # Constructor
        def __init__(self, in_dim, out_dim, function, deriv_function):
            # weight
            self.W = np.random.uniform(low=-1.00, high=1.00, size=(in_dim, out_dim)).astype("float32")
            # bias
            self.b = np.zeros(out_dim).astype("float32")
            # activation function
            self.function = function
            self.deriv_function = deriv_function
            self.u = None
            self.delta = None

        # Forward Propagation
        def f_prop(self, x):
            self.u = np.dot(x, self.W) + self.b
            self.z = self.function(self.u)
            return self.z

        # Back Propagation
        def b_prop(self, delta, W):
            self.delta = self.deriv_function(self.u) * np.dot(delta, W.T)
            return self.delta

    # functions
    def sigmoid(x):
        # return 1 / (1 + np.exp(-x))
        return sp.special.expit(x)

    def deriv_sigmoid(x):
        return sigmoid(x) * (1 - sigmoid(x))

    def liner(x):
        return x

    def deriv_liner(x):
        return np.ones(len(x))

    def f_props(layers, x):
        z = x
        for layer in layers:
            z = layer.f_prop(z)
        return z

    def b_props(layers, delta):
        for i, layer in enumerate(layers[::-1]):
            if i == 0:
                layer.delta = delta
            else:
                delta = layer.b_prop(delta, _W)
            _W = layer.W

    # Definition of Layers
    if num_layers == 3:
        layers = [Layer(2, 8, sigmoid, deriv_sigmoid),
                  Layer(8, 1, liner, deriv_liner)]
    elif num_layers == 4:
        layers = [Layer(2, 16, sigmoid, deriv_sigmoid),
                  Layer(16, 8, sigmoid, deriv_sigmoid),
                  Layer(8, 1, liner, deriv_liner)]
    else:
        layers = [Layer(2, 4, sigmoid, deriv_sigmoid),
                  Layer(4, 1, liner, deriv_liner)]

    def train(X, t, eps=1.0):
        # Forward Propagation
        y = f_props(layers, X)

        delta = y - t

        # Back Propagation
        b_props(layers, delta)

        # Update Parameters
        z = X
        for i, layer in enumerate(layers):
            dW = np.dot(z.T, layer.delta)
            db = np.dot(np.ones(len(z)), layer.delta)

            layer.W = layer.W - eps * dW
            layer.b = layer.b - eps * db

            z = layer.z
        y = f_props(layers, X)

    def test(X):
        y = f_props(layers, X)
        return y

    # record MSE
    # result = np.empty((0, 2))

    # Epoch
    for epoch in range(100):
        # Online Learning
        for x, y in zip(train_X, train_y):
            train(x[np.newaxis, :], y)
        # record MSE
        # pred_y = test(test_X)
        # mse = mean_squared_error(pred_y, test_y)
        # result = np.append(result, np.array([[epoch+1, mse]]), axis=0)
    pred_y = test(test_X)

    # record MSE
    # plt.plot(result[:, 0], result[:, 1])
    # plt.show()

    mse = mean_squared_error(pred_y, test_y)
    print(mse)

    # plot results
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter3D(test_X[:, 0], test_X[:, 1], pred_y)
    file_name = test_function+str(number)+str(num_layers)+'.png'
    plt.savefig(file_name)

    return mse

if __name__ == '__main__':
    # options
    parser = ArgumentParser()
    parser.add_argument(
        '-f', '--function',
        type=str,
        dest='function',
        help='Test Function',
        default='sphere'
    )
    parser.add_argument(
        '-n', '--number',
        type=int,
        dest='number',
        help='Number of dataset',
        default=10000
    )
    parser.add_argument(
        '-l', '--layers',
        type=int,
        dest='layers',
        help='Number of layers',
        default=3
    )
    args = parser.parse_args()
    test_function = args.function
    number = args.number
    num_layers = args.layers
    predict(test_function, number, num_layers)
