# NeuroFox
<img src="assets/logo.png" alt="Logo" width="300" height="300">

> **Note:** For the Turkish version of this document, refer to [README_TR.md](README_TR.md).

## 📄 Table of Contents

1. [Data Generation Functions](#data-generation-functions)
2. [Neural Network Layers](#neural-network-layers)
3. [Activation Functions](#activation-functions)
4. [Regularization Layers](#regularization-layers)
5. [Dense Layer](#dense-layer)
6. [Loss Functions](#loss-functions)
7. [Neural Network Class](#neural-network-class)
8. [Optimizers](#optimizers)
9. [Learning Rate Scheduler](#learning-rate-scheduler)
10. [Utilities](#utilities)
11. [Example Usage](#example-usage)

---

## 📂 Project Summary

This project provides a neural network application that includes various neural network components and optimization techniques. It performs performance analyses using binary classification and various activation functions.

## 📂 Project File Structure

The project has the following file structure:

```
NeuroFox/
│
├── assets/
│   ├── linear_activation.png         # Graph of the linear activation function
│   ├── logo.png                      # Project logo
│   ├── relu_activation.png           # Graph of the ReLU activation function
│   ├── sigmoid_activation.png        # Graph of the sigmoid activation function
│   └── softmax_activation.jpg        # Graph of the softmax activation function
│
├── data/
│   ├── __init__.py                   # File defining the data module
│   └── data.py                       # File containing data generation functions
│
├── layers/
│   ├── __init__.py                   # File defining the layers module
│   ├── dense.py                      # File containing the Dense layer class
│   ├── dropout.py                    # File containing the Dropout regularization layer
│   ├── layer.py                      # File containing the base layer class
│   └── activations/                  # Activation functions
│       ├── __init__.py               # File defining the activations module
│       ├── linear.py                 # File containing the linear activation function
│       ├── relu.py                   # File containing the ReLU activation function
│       ├── sigmoid.py                # File containing the sigmoid activation function
│       └── softmax.py                # File containing the softmax activation function
│
├── losses/
│   ├── __init__.py                   # File defining the losses module
│   ├── binary_crossentropy.py        # File containing the binary cross-entropy loss function
│   ├── binary_focal_loss.py          # File containing the binary focal loss function
│   └── categorical_crossentropy.py   # File containing the categorical cross-entropy loss function
│
├── neural_network/
│   ├── __init__.py                   # File defining the neural_network module
│   └── neural_network.py             # File defining the neural network structure
│
├── optimizers/
│   ├── __init__.py                   # File defining the optimizers module
│   ├── adagrad_optimizer.py          # File containing the Adagrad optimization algorithm
│   ├── adam_optimizer.py             # File containing the Adam optimization algorithm
│   ├── learning_rate_scheduler.py    # File containing the learning rate scheduler
│   ├── rmsprop_optimizer.py          # File containing the RMSprop optimization algorithm
│   └── sgd_optimizer.py              # File containing the Stochastic Gradient Descent (SGD) optimization algorithm
│
├── utils/
│   ├── __init__.py                   # File defining the utils module
│   ├── binary_classification.py      # Tools for generating binary classification data
│   ├── model_utils.py                # Various utility functions related to models
│   ├── one_hot_encoding.py           # One-hot encoding function
│   ├── standard_scaler.py            # Function for standardizing data
│   └── train_test_split.py           # Function for splitting data into training and testing sets
│
├── binary_classification_model.py    # Example of a binary classification model
├── iris_dataset_model.py             # Example model with the IRIS dataset
├── xor_model.py                      # Example model with the XOR dataset
└── README.md                         # General information about the project, installation, and usage instructions
```

### File Descriptions

- **`assets/`**: Directory containing visual files related to the project. This directory includes visual representations of activation function formulas.
- **`data/`**: Files containing functions necessary for generating and testing training data.
- **`layers/`**: Files containing neural network layers and activation functions. Details of activation functions are also found here.
- **`losses/`**: Loss functions and their implementations.
- **`neural_network/`**: Files defining the building blocks of the neural network model.
- **`optimizers/`**: Files containing different optimization algorithms and learning rate schedulers.
- **`utils/`**: Functions for data processing, model management, and other utility tools.
- **`README.md`**: General information about the project, installation instructions, usage details, and examples.

### 1. **Data Generation Functions**

#### **`create_xor_data(num_samples)`**
Generates XOR data for binary classification tasks.

- **Usage**:
  ```python
  X, y = create_xor_data(1000)
  ```

- **Parameters**:
  -  `num_samples` (int): Number of data points to generate.
- **Returns**: 
  - `X`: Input features
  - `y`: Labels

#### **`create_binary_classification_data(num_samples=1000)`**
Generates binary classification data with an option to add noise.

- **Usage**:
  ```python
  X, y = create_binary_classification_data(samples=1000, noise=0.1)
  ```

- **Parameters**:
  - `num_samples` (int): Number of data points to generate.
- **Returns**: 
  - `X`: Input features
  - `y`: Labels

### 2. **Neural Network Layers**

#### **`Layer`**
Base class for all layers within the neural network.

### 3. **Activation Functions**

#### **`ActivationSoftmax`**
Applies the Softmax activation function to the input.

- **Softmax Formula**:
  - $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

- <img src="assets/softmax_activation.jpg" alt="Softmax" width="500" height="300">

#### **`ActivationSigmoid`**
Applies the Sigmoid activation function to the input.

- **Sigmoid Formula**:
  -  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- <img src="assets/sigmoid_activation.png" alt="Sigmoid" width="500" height="300">

#### **`ActivationReLU`**
Applies the ReLU activation function to the input.

- **ReLU Formula**:
  -  $$\text{ReLU}(x) = \max(0, x)$$

- <img src="assets/relu_activation.png" alt="ReLu" width="500" height="300">

#### **`ActivationLinear`**
Applies the linear activation function to the input (no change).

- **Linear Formula**:
  - $$f(x) = x$$

- <img src="assets/linear_activation.png" alt="Linear" width="500" height="300">

### 4. **Regularization Layers**

#### **`Dropout(rate=0.5)`**
Applies dropout regularization.

- **Usage**: 
  ```python
  dropout_layer = Dropout(rate=0.5)
  ```

- **Parameters**:
  - `rate` (float): The proportion of input units to drop.

### 5. **Dense Layer**

#### **`Dense(input_size, output_size)`**
A fully connected layer in the neural network.

- **Usage**: 
  ```python
  dense_layer = Dense(input_size=128, output_size=64)
  ```

- **Parameters**:
  - `input_size` (int): Number of input features.
  - `output_size` (int): Number of output features.

### 6. **Loss Functions**

#### **`BinaryCrossentropy`**
Calculates binary cross-entropy loss.

- **Formula**:
  -  $$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

#### **`CategoricalCrossentropy`**
Calculates categorical cross-entropy loss.

- **Formula**:
  -  $$L = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$

#### **`BinaryFocalLoss(gamma=2, alpha=0.25)`**
Calculates binary focal loss, often used to address class imbalance.

- **Formula**:
  - $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

- **Parameters**:
  - `gamma` (float): Focusing parameter.
  - `alpha` (

float): Balancing parameter.

### 7. **Neural Network Class**

#### **`NeuralNetwork`**
Defines the architecture and functionality of the neural network.

- **Usage**:
  ```python
  nn = NeuralNetwork()
  nn.add_layer(Dense(input_size=128, output_size=64))
  nn.add_layer(ActivationReLU())
  nn.compile(loss=BinaryCrossentropy(), optimizer=AdamOptimizer())
  nn.fit(X_train, y_train, epochs=10)
  ```

### 8. **Optimizers**

#### **`AdamOptimizer(learning_rate=0.001)`**
Implements the Adam optimization algorithm.

- **Parameters**:
  - `learning_rate` (float): Learning rate.

#### **`SGDOptimizer(learning_rate=0.01)`**
Implements the Stochastic Gradient Descent (SGD) optimization algorithm.

- **Parameters**:
  - `learning_rate` (float): Learning rate.

#### **`AdagradOptimizer(learning_rate=0.01)`**
Implements the Adagrad optimization algorithm.

- **Parameters**:
  - `learning_rate` (float): Learning rate.

#### **`RMSPropOptimizer(learning_rate=0.001, rho=0.9)`**
Implements the RMSProp optimization algorithm.

- **Parameters**:
  - `learning_rate` (float): Learning rate.
  - `rho` (float): Decay factor.

### 9. **Learning Rate Scheduler**

#### **`LearningRateScheduler(initial_lr=0.01, schedule=lambda epoch: 0.01)`**
Adjusts the learning rate during training.

- **Parameters**:
  - `initial_lr` (float): Initial learning rate.
  - `schedule` (function): Function that defines how the learning rate changes over epochs.

### 10. **Utilities**

#### **`one_hot_encode(labels, num_classes)`**
Converts integer labels to one-hot encoded format.

- **Usage**:
  ```python
  one_hot_labels = one_hot_encode(labels, num_classes=3)
  ```

- **Parameters**:
  - `labels` (array-like): Integer labels.
  - `num_classes` (int): Total number of classes.

#### **`standardize_data(X)`**
Standardizes the dataset.

- **Usage**:
  ```python
  standardized_X = standardize_data(X)
  ```

#### **`train_test_split(X, y, test_size=0.2)`**
Splits data into training and testing sets.

- **Usage**:
  ```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  ```

### 11. **Example Usage**

#### **Binary Classification Example**

```python
from neural_network import NeuralNetwork
from layers import Dense, ActivationReLU
from losses import BinaryCrossentropy
from optimizers import AdamOptimizer
from utils import create_binary_classification_data, train_test_split

# Generate data
X, y = create_binary_classification_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train model
model = NeuralNetwork()
model.add_layer(Dense(input_size=2, output_size=8))
model.add_layer(ActivationReLU())
model.add_layer(Dense(input_size=8, output_size=1))
model.compile(loss=BinaryCrossentropy(), optimizer=AdamOptimizer())
model.fit(X_train, y_train, epochs=10)

# Evaluate model
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

#### **IRIS Dataset Example**

```python
from neural_network import NeuralNetwork
from layers import Dense, ActivationSoftmax
from losses import CategoricalCrossentropy
from optimizers import AdamOptimizer
from utils import load_iris_data, train_test_split

# Load data
X, y = load_iris_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train model
model = NeuralNetwork()
model.add_layer(Dense(input_size=4, output_size=32))
model.add_layer(ActivationReLU())
model.add_layer(Dense(input_size=32, output_size=3))
model.add_layer(ActivationSoftmax())
model.compile(loss=CategoricalCrossentropy(), optimizer=AdamOptimizer())
model.fit(X_train, y_train, epochs=10)

# Evaluate model
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

#### **XOR Dataset Example**

```python
from neural_network import NeuralNetwork
from layers import Dense, ActivationReLU
from losses import BinaryCrossentropy
from optimizers import AdamOptimizer
from utils import create_xor_data, train_test_split

# Generate data
X, y = create_xor_data(1000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train model
model = NeuralNetwork()
model.add_layer(Dense(input_size=2, output_size=8))
model.add_layer(ActivationReLU())
model.add_layer(Dense(input_size=8, output_size=1))
model.compile(loss=BinaryCrossentropy(), optimizer=AdamOptimizer())
model.fit(X_train, y_train, epochs=10)

# Evaluate model
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
