import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
a = np.loadtxt('alior.txt')

y = a[:30,[0]]
c = a[:30,1:]

cTraining, cTest, yTraining, yTest = train_test_split(c, y, test_size=0.3)
v = np.linalg.pinv(cTraining) @ yTraining
print(v)

eTraining = sum((yTraining - (v[0] * cTraining[:, [0]] + v[1] * cTraining[:, [1]] + v[2] * cTraining[:, [2]])) ** 2) / cTraining.shape[0]
eTest = sum((yTest - (v[0] * cTest[:, [0]] + v[1] * cTest[:, [1]] + v[2] * cTest[:, [2]])) ** 2) / cTest.shape[0]

print("Error Training: ", eTraining)
print("Error Test: ", eTest)

plt.plot(yTest, 'r-')
plt.plot(v[0] * cTest[:, [0]] + v[1] * cTest[:, [1]] + v[2] * cTest[:, [2]])
plt.show()