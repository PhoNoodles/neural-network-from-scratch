# Forward propagation
def forward(x, w1, w2):
    hidden = x * w1
    predicted = hidden * w2

    return hidden, predicted


# Loss function
def squared_error(predicted, actual):
    error = predicted - actual
    loss = error ** 2

    return loss


# Derivative of loss
def squared_error_derivative(predicted, actual):
    difference = predicted - actual

    return 2 * difference


# Backpropagation - weight gradient
def weight_gradient(incoming_gradient, neuron_input):
    return incoming_gradient * neuron_input


# Backpropagation - input gradient
def input_gradient(incoming_gradient, weight):
    return incoming_gradient * weight


# Gradient descent
def update_weight(weight, gradient, learning_rate):
    change = gradient * learning_rate
    new_weight = weight - change

    return new_weight


# Training data
x = 2
actual = 20

# Parameters
w1 = 3.0
w2 = 4.0

# Hyperparameters
learning_rate = 0.005
epochs = 100


# Training loop
for epoch in range(epochs):

    # 1. Forward propagation
    hidden, predicted = forward(x, w1, w2)

    # 2. Calculate loss
    loss = squared_error(predicted, actual)

    # 3. Backpropagation
    loss_gradient = squared_error_derivative(predicted, actual)

    gradient_w2 = weight_gradient(loss_gradient, hidden)

    gradient_hidden = input_gradient(loss_gradient, w2)

    gradient_w1 = weight_gradient(gradient_hidden, x)

    # 4. Gradient descent
    w1 = update_weight(w1, gradient_w1, learning_rate)
    w2 = update_weight(w2, gradient_w2, learning_rate)

    print("Epoch:", epoch, "Loss:", loss)


# Final prediction after training
hidden, predicted = forward(x, w1, w2)

print("Final prediction:", predicted)
print("Final weight 1:", w1)
print("Final weight 2:", w2)