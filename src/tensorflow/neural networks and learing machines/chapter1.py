import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def moon(r, w, n, d, show):
    inner_rad = r - w / 2
    rho = np.ones(n) * inner_rad + np.random.rand(n) * w
    theta = np.random.rand(n) * np.pi
    x_up = rho * np.array([np.cos(theta), np.sin(theta)])
    y_up = np.ones(n)

    x_down = rho * np.array([np.cos(theta), -np.sin(theta)]) + \
        (np.ones(x_up.shape).T * np.array([inner_rad, -d])).T
    y_down = -np.ones(n)

    if show:
        fig = plt.figure()
        ax = fig.gca()
        ax.set_aspect('equal')
        plt.scatter(x_up[0, :], x_up[1, :], marker='+')
        plt.scatter(x_down[0, :], x_down[1, :], marker='+')
        plt.show()

    return np.concatenate((x_up, x_down)), np.concatenate((y_up, y_down))


def input():
    data, labels = moon(10, 6, 3000, 1, True)
    return data, labels


def inference(data):
    w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    b = tf.Variable()
    return w, b


def loss(logits, labels):
    pass


def train(loss):
    pass


def evaluation(logits, labels):
    pass


def main():
    data, labels = input()


if __name__ == '__main__':
    main()
