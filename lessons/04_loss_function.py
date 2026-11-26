def squared_error(predicted, actual):
    error = predicted - actual
    loss = error ** 2

    return loss


predicted = 1.85
actual = 3

loss = squared_error(predicted, actual)

print(loss)


def mean_squared_error(predictions, actuals):
    if len(predictions) == 0:
        raise ValueError("Predictions cannot be empty")
    if len(predictions) != len(actuals):
        raise ValueError("need same prediction and actuals")
    total = 0

    for i in range(len(predictions)):
        total += squared_error(predictions[i],actuals[i])

    total = total / len(predictions)

    return total

predictions = [2, 5, 3, 8]
actuals = [3, 4, 5, 6]

loss = mean_squared_error(predictions, actuals)

print(loss)
