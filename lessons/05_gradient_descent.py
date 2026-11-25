def update_weight(weight, gradient, learning_rate):

    change = gradient * learning_rate

    new_weight = weight - change

    return new_weight


weight = 3
gradient = 64
learning_rate = 0.01

new_weight = update_weight(weight, gradient, learning_rate)

print(new_weight)