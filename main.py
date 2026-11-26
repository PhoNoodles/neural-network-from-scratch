from src.network import forward, train


def main():
    training_data = [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]
    w1, w2, history = train(training_data, 3.0, 4.0)
    print("Initial epoch loss:", history[0])
    print("Final epoch loss:", history[-1])
    print("Learned weights:", w1, w2)
    for x in range(6, 11):
        hidden, predicted = forward(x, w1, w2)
        print("Input:", x, "Prediction:", predicted)


if __name__ == "__main__":
    main()
