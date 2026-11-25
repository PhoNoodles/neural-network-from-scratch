def neuron(inputs, weights, bias):

    if len(inputs) != len(weights):
        raise ValueError("Inputs and weights must have the same length")
    total = 0

    for i in range(len(inputs)):
        val = inputs[i] * weights[i]

        total += val

    total += bias
    return total

def relu(z):
    if z < 0:
        return 0
    return z

inputs = [2, 3]

weights = [
    [0.5, -1],
    [1, 0.5],
    [-0.5, 2]
]

biases = [1, 0, -1]

def layer(inputs, weights, biases):
    if len(weights) != len(biases):
        raise ValueError("Each neuron must have a bias")
    outputs = []
    for i in range(len(weights)):
        output = neuron(inputs,weights[i],biases[i])
        activated_output = relu(output)
        outputs.append(activated_output)


    return outputs


hidden_outputs = layer(inputs, weights, biases)

print(hidden_outputs)


output_weights = [
    [0.5, 0.3, 0.2]
]

output_biases = [0]

final_output = layer(
    hidden_outputs,
    output_weights,
    output_biases
)

print(final_output)