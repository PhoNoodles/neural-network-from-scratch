# Neural Network from Scratch

A learning project implementing neural-network fundamentals in plain Python, without NumPy, PyTorch or TensorFlow.

## Run

From the repository root with Python 3:

```bash
python main.py
python -m unittest discover -s tests -v
```

No third-party packages are required.

## Learning roadmap

1. Artificial neuron: weighted sum and bias.
2. ReLU activation.
3. Dense layers and forward propagation.
4. Squared error and mean squared error (originally included in lesson 03).
5. Gradient descent.
6. Backpropagation using the chain rule.
7. Training on one example.
8. Fixed-order SGD on multiple examples.
9. Reusable modules and numerical gradient checks.
10. Next: trainable nonlinear dense layers and XOR.

The original standalone lesson files remain learning references. Lesson 01 includes a corrected valid example and an explicit invalid-input demonstration.

## Architecture

- `src/neuron.py`: weighted sums.
- `src/activations.py`: ReLU and its derivative (zero at the kink by convention).
- `src/layers.py`: dense forward pass.
- `src/losses.py`: losses and squared-error derivative.
- `src/backpropagation.py`: scalar linear-operation gradients.
- `src/optimizers.py`: gradient descent.
- `src/network.py`: the two-weight training example.
- `main.py`: train and predict on unseen inputs.
- `tests/`: standard-library tests, including finite-difference gradient checks.

## Training cycle

Forward pass -> loss -> backpropagation -> parameter update -> repeat.

The current trainable model is deliberately linear: predicted = x * w1 * w2. It learns the synthetic relationship y = 10x, not arbitrary nonlinear patterns. The dense layer supports ReLU for forward propagation, but general nonlinear backpropagation and bias training are future lessons. Epoch loss is the average of losses encountered while weights change, not a fixed-model evaluation.

The next milestone is learning XOR, followed by a reusable nonlinear network and eventually handwritten digit recognition.
