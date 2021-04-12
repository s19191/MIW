import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from plotka import plot_decision_regions

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class MultiClassPerceptron:
    def __init__(self, ppn1, ppn2):
        self.ppn1 = ppn1
        self.ppn2 = ppn2

    def predict(self, X):
        return np.where(self.ppn1.predict(X) == 1, 0,
                        np.where(self.ppn2.predict(X) == 1, 2, 1))


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def getProbability(self, y):
        return self.activation(self.net_input(y))

class MultiClassLogisticRegressionGD:
    def __init__(self, logisticRegression1, logisticRegression2):
        self.lrgd1 = logisticRegression1
        self.lrgd2 = logisticRegression2

    def predict(self, X):
        return np.where(self.lrgd1.predict(X) == 1, 0,
                        np.where(self.lrgd2.predict(X) == 1, 2, 1))

def main():

    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    X_train_01_subset = np.copy(y_train)
    X_train_01_subset = X_train_01_subset[(X_train_01_subset != 2)]
    y_train_01_subset = np.copy(X_train)
    y_train_01_subset = y_train_01_subset[(y_train != 2)]

    X_train_01_subset[(X_train_01_subset != 0)] = -1
    X_train_01_subset[(X_train_01_subset == 0)] = 1

    ppn01 = Perceptron(eta=0.1, n_iter=10)
    ppn01.fit(y_train_01_subset, X_train_01_subset)

    X_train_02_subset = np.copy(y_train)
    X_train_02_subset = X_train_02_subset[(X_train_02_subset != 0)]
    y_train_02_subset = np.copy(X_train)
    y_train_02_subset = y_train_02_subset[(y_train != 0)]

    X_train_02_subset[(X_train_02_subset != 2)] = -1
    X_train_02_subset[(X_train_02_subset == 2)] = 1

    ppn02 = Perceptron(eta=0.1, n_iter=1000)
    ppn02.fit(y_train_02_subset, X_train_02_subset)

    mppn = MultiClassPerceptron(ppn01, ppn02)

    plot_decision_regions(X=X_test, y=y_test, classifier=mppn)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

    X_train_01_subset[(X_train_01_subset != 1)] = 0
    lrgd01 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd01.fit(y_train_01_subset, X_train_01_subset)

    print('Probability of first regresion')
    probabilities01 = lrgd01.getProbability(y_train_01_subset)
    for each in probabilities01:
        print('%1.7f' % each, end=" ")
    print()

    X_train_02_subset[(X_train_02_subset != 1)] = 0
    lrgd02 = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    lrgd02.fit(y_train_02_subset, X_train_02_subset)

    print('Probability of second regresion')
    probabilities02 = lrgd02.getProbability(y_train_02_subset)
    for each in probabilities02:
        print('%1.7f' % each, end=" ")
    print()

    mlrgd = MultiClassLogisticRegressionGD(lrgd01, lrgd02)

    plot_decision_regions(X=X_test, y=y_test, classifier=mlrgd)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    main()