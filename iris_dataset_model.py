import numpy as np
from sklearn.datasets import load_iris
from utils import train_test_split, StandardScaler, one_hot_encoding
from neural_network import NeuralNetwork
from layers import Dense, ActivationReLU, ActivationSoftmax, Dropout
from losses import CategoricalCrossentropy
from optimizers import RMSpropOptimizer


data = load_iris()
X = data.data
y = data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_one_hot = one_hot_encoding(y, num_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.05, random_state=42)

model = NeuralNetwork()

model.add_layer(Dense(input_size=X_train.shape[1], output_size=8))
model.add_activation(ActivationReLU())
model.add_dropout(Dropout(rate=0.5))

model.add_layer(Dense(input_size=8, output_size=3))
model.add_activation(ActivationSoftmax())
model.set_batch_size(7)

loss_function = CategoricalCrossentropy()
optimizer = RMSpropOptimizer(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss_function=loss_function)

history = model.fit(X_train, y_train, epochs=800, print_loss_acc=True, return_history=True)

predictions = model.predict(X_test)
predictions_class = np.argmax(predictions, axis=1)
y_test_class = np.argmax(y_test, axis=1)
correct_predictions = np.sum(predictions_class == y_test_class)
accuracy = correct_predictions / y_test_class.shape[0]
print('Accuracy:', accuracy * 100)

model.summary()
