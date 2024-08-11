from neural_network import NeuralNetwork
from layers import Dense, ActivationReLU, ActivationSigmoid
from losses import BinaryCrossentropy
from optimizers import AdamOptimizer
from data import create_binary_classification_data
from utils import train_test_split, classify

X, y = create_binary_classification_data(100)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = NeuralNetwork()
nn.add_layer(Dense(2, 4))
nn.add_activation(ActivationReLU())
nn.add_layer(Dense(4, 1))
nn.add_activation(ActivationSigmoid())

loss_function = BinaryCrossentropy()
optimizer = AdamOptimizer(learning_rate=0.001)

nn.compile(optimizer=optimizer, loss_function=loss_function)

nn.fit(x_train, y_train, epochs=1000, print_loss_acc=True)

output = nn.predict(x_test)
accuracy = loss_function.accuracy(output, y_test)

output = classify(output)

print("Final output:")
print(output[:5])
print("True labels:")
print(y_test[:5])
print("Final accuracy:", accuracy)
