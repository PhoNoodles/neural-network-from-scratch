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


# Loss derivative
def squared_error_derivative(predicted, actual):
    difference = predicted - actual

    return 2 * difference


# Backpropagation
def weight_gradient(incoming_gradient, neuron_input):
    return incoming_gradient * neuron_input


def input_gradient(incoming_gradient, weight):
    return incoming_gradient * weight


# Gradient descent
def update_weight(weight, gradient, learning_rate):
    change = gradient * learning_rate
    new_weight = weight - change

    return new_weight


# Training dataset
training_data = [
    (1, 10),
    (2, 20),
    (3, 30),
    (4, 40),
    (5, 50)
]

# Parameters
w1 = 3.0
w2 = 4.0

# Hyperparameters
learning_rate = 0.0001
epochs = 100


# Training loop
for epoch in range(epochs):

    total_loss = 0

    for x, actual in training_data:

        # Forward propagation
        hidden, predicted = forward(x, w1, w2)

        # Calculate loss
        loss = squared_error(predicted, actual)
        total_loss += loss

        # Backpropagation
        loss_gradient = squared_error_derivative(predicted, actual)

        gradient_w2 = weight_gradient(loss_gradient, hidden)

        gradient_hidden = input_gradient(loss_gradient, w2)

        gradient_w1 = weight_gradient(gradient_hidden, x)

        # Gradient descent
        w1 = update_weight(w1, gradient_w1, learning_rate)
        w2 = update_weight(w2, gradient_w2, learning_rate)

    average_loss = total_loss / len(training_data)

    print("Epoch:", epoch, "Average loss:", average_loss)


# Inference on unseen data
print("\nTesting trained model:")

test_inputs = [6, 7, 8, 9, 10]

for x in test_inputs:
    hidden, predicted = forward(x, w1, w2)

    print("Input:", x, "Prediction:", predicted)