from neural_network import NeuralNetwork
from layers import ActivationReLU, ActivationSigmoid, Dense
from optimizers import AdamOptimizer, LearningRateScheduler
from losses import BinaryCrossentropy
from data import create_xor_data
from utils import save_model, load_model, train_test_split

X, y = create_xor_data(1000)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nn = NeuralNetwork()
nn.add_layer(Dense(2, 4))
nn.add_activation(ActivationReLU())
nn.add_layer(Dense(4, 1))
nn.add_activation(ActivationSigmoid())

loss_function = BinaryCrossentropy()
optimizer = AdamOptimizer(learning_rate=0.003)
scheduler = LearningRateScheduler(initial_lr=0.001, decay_factor=0.5, decay_every=10)

nn.compile(optimizer=optimizer, loss_function=loss_function, scheduler=scheduler)

history = nn.fit(x_train, y_train, epochs=1000, return_history=True)

save_model(nn, 'model.pkl')
nn = load_model('model.pkl')

l, a = nn.evaluate(x_test, y_test)
print("Final loss:", l)
print("Final accuracy:", a)

nn.summary()
