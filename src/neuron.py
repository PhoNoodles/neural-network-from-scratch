def neuron(inputs, weights, bias):
    """Calculate a neuron's weighted sum before activation."""
    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    total = bias
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    return total
