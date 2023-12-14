from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import classification_report

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype("float32") / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype("float32") / 255

y_train_binary = (y_train >= 5).astype(int)
y_test_binary = (y_test >= 5).astype(int)

# Build LeNet model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train, y_train_binary, epochs=5, batch_size=64, validation_split=0.2)

# Evaluate the model
y_pred_prob = model.predict(x_test)
y_pred = (y_pred_prob >= 0.5).astype(int)
print(classification_report(y_test_binary, y_pred))
