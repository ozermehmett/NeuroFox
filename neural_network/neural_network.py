import numpy as np
from layers import ActivationLinear


class NeuralNetwork:
    def __init__(self):
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        self.layers = []
        self.activations = []
        self.dropouts = []
        self.layer_outputs = []
        self.activation_outputs = []
        self.batch_size = 32

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_activation(self, activation=None):
        if activation is None:
            activation = ActivationLinear()
        self.activations.append(activation)

    def add_dropout(self, dropout):
        self.dropouts.append(dropout)

    def forward(self, X, training=True):
        self.layer_outputs = []
        self.activation_outputs = []

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].forward(X)
            else:
                self.layers[i].forward(self.activation_outputs[i - 1])

            self.layer_outputs.append(self.layers[i].output)

            if i < len(self.activations):
                self.activations[i].forward(self.layers[i].output)
                self.activation_outputs.append(self.activations[i].output)

            if i < len(self.dropouts):
                self.activation_outputs[-1] = self.dropouts[i].forward(self.activation_outputs[-1], training)

        return self.activation_outputs[-1]

    def backward(self, dvalues):
        for i in reversed(range(len(self.layers))):
            if i < len(self.dropouts):
                dvalues = self.dropouts[i].backward(dvalues)

            if i < len(self.activations):
                self.activations[i].backward(dvalues)

            self.layers[i].backward(self.activations[i].dinputs if i < len(self.activations) else dvalues)
            dvalues = self.layers[i].dinputs

    def update_params(self, learning_rate):
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases

    def compile(self, optimizer, loss_function, scheduler=None):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler

    def fit(self, x_train, y_train, epochs, print_loss_acc=False, return_history=False):
        history = {'loss': [], 'accuracy': []}
        num_samples = x_train.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                X_batch = x_train[start:end]
                y_batch = y_train[start:end]

                output = self.forward(X_batch)
                loss = self.loss_function.forward(output, y_batch)
                accuracy = self.loss_function.accuracy(output, y_batch)

                dvalues = self.loss_function.backward()
                self.backward(dvalues)

                for layer in self.layers:
                    self.optimizer.update(layer)

                epoch_loss += loss * len(y_batch)
                epoch_accuracy += accuracy * len(y_batch)

            epoch_loss /= num_samples
            epoch_accuracy /= num_samples

            if self.scheduler:
                self.optimizer.learning_rate = self.scheduler.get_learning_rate(epoch)

            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_accuracy)

            if print_loss_acc:
                print(f'Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')

        if return_history:
            return history

    def predict(self, x):
        output = self.forward(x, training=False)
        return output

    def evaluate(self, x_test, y_test):
        output = self.forward(x_test, training=False)

        loss = self.loss_function.forward(output, y_test)
        accuracy = self.loss_function.accuracy(output, y_test)

        return round(loss, 4), accuracy

    def summary(self):
        sep_size = 85
        print("=" * sep_size)
        print(" " * 35 + "Model Summary" + " " * 35)
        print("=" * sep_size)

        print(f"{'Layer':<10} {'Type':<20} {'Input Size':<15} {'Output Size':<15} {'Activation':<20}")
        print("=" * sep_size)

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            print(f"{f'Layer {i + 1}':<10} {layer.__class__.__name__:<20} "
                  f"{layer.input_size:<15} {layer.output_size:<15}"
                  f" {activation.__class__.__name__:<20}")

        print("=" * sep_size)
        print(f"Total layers: {len(self.layers)}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Loss Function: {self.loss_function.__class__.__name__}")
        if self.scheduler:
            print(f"Learning Rate Scheduler: {self.scheduler.__class__.__name__}")
        else:
            print("Learning Rate Scheduler: None")
        print(f"Batch Size: {self.batch_size}")
        print("=" * sep_size)
