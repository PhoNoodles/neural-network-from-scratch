from .losses import squared_error, squared_error_derivative
from .backpropagation import weight_gradient, input_gradient
from .optimizers import update_weight


def forward(x, w1, w2):
    """Two scalar linear layers, without biases or activations."""
    hidden = x * w1
    predicted = hidden * w2
    return hidden, predicted


def gradients(x, actual, w1, w2):
    hidden, predicted = forward(x, w1, w2)
    loss_gradient = squared_error_derivative(predicted, actual)
    gradient_w2 = weight_gradient(loss_gradient, hidden)
    gradient_hidden = input_gradient(loss_gradient, w2)
    gradient_w1 = weight_gradient(gradient_hidden, x)
    return gradient_w1, gradient_w2


def train(training_data, w1, w2, learning_rate=0.0001, epochs=100):
    """Fixed-order SGD; report mean of losses encountered each epoch."""
    if not training_data:
        raise ValueError("Training data cannot be empty")
    if learning_rate <= 0 or epochs < 1:
        raise ValueError("Learning rate and epochs must be positive")
    history = []
    for epoch in range(epochs):
        total_loss = 0
        for x, actual in training_data:
            hidden, predicted = forward(x, w1, w2)
            total_loss += squared_error(predicted, actual)
            gradient_w1, gradient_w2 = gradients(x, actual, w1, w2)
            w1 = update_weight(w1, gradient_w1, learning_rate)
            w2 = update_weight(w2, gradient_w2, learning_rate)
        history.append(total_loss / len(training_data))
    return w1, w2, history
