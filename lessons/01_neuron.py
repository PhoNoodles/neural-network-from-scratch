def neuron(inputs, weights, bias):

    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    total = 0

    for i in range(len(inputs)):
        val = inputs[i] * weights[i]

        total += val

    total += bias
    return total


inputs = [2, 3, 4]
weights = [0.5, -1]
bias = 1

output = neuron(inputs, weights, bias)

print(output)