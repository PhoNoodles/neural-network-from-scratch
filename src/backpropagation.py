def weight_gradient(incoming_gradient, neuron_input):
    """Gradient of a weight in a linear operation."""
    return incoming_gradient * neuron_input


def input_gradient(incoming_gradient, weight):
    """Gradient passed to the input of a linear operation."""
    return incoming_gradient * weight
