from time import sleep
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-2.0, 2.0, num=41).reshape(1, 41)
Y = np.power(X, 2) + 1 * (np.random.rand(X.size) - 0.5)

# siec

S1 = 4
W1 = np.random.rand(S1, 1) - 0.5
B1 = np.random.rand(S1, 1) - 0.5
W2 = np.random.rand(1, S1) - 0.5
B2 = np.random.rand(1, 1) - 0.5

lr = 0.01

for epoka in range(1, 200):

    # odpowiedz sieci
    s = W1 @ X + B1 @ np.ones(41).reshape(1, 41)
    A1 = np.arctan(s)
    A2 = W2 @ A1 + B2

    # propagacja wsteczna
    E2 = Y - A2
    E1 = W2.T @ E2

    dW2 = lr * E2 @ A1.T
    dB2 = lr * E2 @ np.ones(E2.size).T
    dW1 = lr * 1. / (1 + s * s) * E1 @ X.T
    dB1 = lr * 1. / (1 + s * s) * E1 @ np.ones(X.size).T
    dB1 = dB1.reshape(4, 1)

    W2 = W2 + dW2
    B2 = B2 + dB2
    W1 = W1 + dW1
    B1 = B1 + dB1

    if epoka % 10 == 0:
        plt.plot(X[0], A2[0])
        plt.plot(X, Y, 'r*')
        plt.show()
        sleep(.5)