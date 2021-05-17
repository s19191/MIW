from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model01 = models.Sequential()
model01.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model01.add(layers.MaxPooling2D((2, 2)))
model01.add(layers.Conv2D(64, (3, 3), activation='relu'))
model01.add(layers.MaxPooling2D((2, 2)))
model01.add(layers.Conv2D(64, (3, 3), activation='relu'))
model01.add(layers.Flatten())
model01.add(layers.Dense(64, activation='relu'))
model01.add(layers.Dense(10, activation='softmax'))
model01.summary()

model01.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model01.fit(train_images, train_labels, epochs=10, batch_size=64)

test_loss, test_acc = model01.evaluate(test_images, test_labels)
print('test_acc = ', test_acc)

model02 = models.Sequential()
model02.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model02.add(layers.MaxPooling2D((2, 2)))
model02.add(layers.Conv2D(64, (3, 3), activation='relu'))
model02.add(layers.MaxPooling2D((2, 2)))
model02.add(layers.Conv2D(64, (3, 3), activation='relu'))
model02.add(layers.Flatten())
model02.add(layers.Dense(64, activation='relu'))
model02.add(layers.Dense(10, activation='softmax'))
model02.summary()

model02.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model02.fit(train_images, train_labels, epochs=10, batch_size=64)

test_loss, test_acc = model02.evaluate(test_images, test_labels)
print('test_acc = ', test_acc)

model03 = models.Sequential()
model03.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model03.add(layers.MaxPooling2D((2, 2)))
model03.add(layers.Conv2D(64, (3, 3), activation='relu'))
model03.add(layers.MaxPooling2D((2, 2)))
model03.add(layers.Conv2D(64, (3, 3), activation='relu'))
model03.add(layers.Flatten())
model03.add(layers.Dense(64, activation='relu'))
model03.add(layers.Dense(10, activation='softmax'))
model03.summary()

model03.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model03.fit(train_images, train_labels, epochs=10, batch_size=64)

test_loss, test_acc = model03.evaluate(test_images, test_labels)
print('test_acc = ', test_acc)