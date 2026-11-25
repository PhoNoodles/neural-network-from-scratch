def weight_gradient(loss_gradient, neuron_input):
    return loss_gradient * neuron_input


def squared_error_derivative(predicted, actual):

    difference = predicted - actual

    return 2 * difference


predicted = 24
actual = 20
hidden = 6

loss_gradient = squared_error_derivative(predicted, actual)

gradient_w2 = weight_gradient(loss_gradient, hidden)

print(gradient_w2)