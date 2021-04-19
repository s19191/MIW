import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('dane10.txt')

x = a[:,[0]]
y = a[:,[1]]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

c = np.hstack([np.abs(X_train), np.ones(X_train.shape)]) # model liniowy z wartością bezwględną
c1 = np.hstack([X_train**2, X_train, np.ones(X_train.shape)]) # model kwadratowy
# c2 = np.hstack([X_train, np.ones(X_train.shape)]) # model liniowy

v = np.linalg.pinv(c) @ y_train
v1 = np.linalg.pinv(c1) @ y_train
# v2 = np.linalg.inv(c2.T @ c2) @ c2.T @ y_train

plt.plot(x, y, 'ro')
plt.plot(x, v[0]*np.abs(x) + v[1])
plt.plot(x, v1[0]*x**2 + v1[1]*x + v1[2])
# plt.plot(x, v[0]*x + v2[1])
plt.show()