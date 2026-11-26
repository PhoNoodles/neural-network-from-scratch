from .neuron import neuron
from .activations import relu


def layer(inputs, weights, biases, activation=relu):
    """One row of weights and one bias per neuron."""
    if len(weights) != len(biases):
        raise ValueError("Each neuron must have a bias")
    outputs = []
    for i in range(len(weights)):
        z = neuron(inputs, weights[i], biases[i])
        outputs.append(activation(z))
    return outputs
