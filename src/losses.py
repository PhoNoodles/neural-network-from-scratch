def squared_error(predicted, actual):
    error = predicted - actual
    return error ** 2


def squared_error_derivative(predicted, actual):
    return 2 * (predicted - actual)


def mean_squared_error(predictions, actuals):
    if not predictions or len(predictions) != len(actuals):
        raise ValueError("Expected nonempty lists of equal length")
    total = 0
    for i in range(len(predictions)):
        total += squared_error(predictions[i], actuals[i])
    return total / len(predictions)
