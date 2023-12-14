import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import classification_report

# Завантаження та підготовка даних Fashion MNIST
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255

# Преобразування міток у категоріальний формат
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Створення кастомної моделі (LeNet-подібна)
model = Sequential()
model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Тренування моделі (варіант: використовуйте менше епох)
model.fit(x_train, y_train_categorical, epochs=20, batch_size=64, validation_split=0.2)

# Оцінка моделі
y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Виведення classification report
print(classification_report(y_test, y_pred))
