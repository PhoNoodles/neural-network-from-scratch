def relu(z):
    if z < 0:
        return 0
    return z


def relu_derivative(z):
    """Use zero as the chosen derivative at the kink."""
    if z <= 0:
        return 0
    return 1
