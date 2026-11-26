def update_weight(weight, gradient, learning_rate):
    change = gradient * learning_rate
    return weight - change
