import unittest

from src.neuron import neuron
from src.activations import relu, relu_derivative
from src.layers import layer
from src.losses import squared_error, squared_error_derivative, mean_squared_error
from src.optimizers import update_weight
from src.network import forward, gradients, train


class FundamentalsTests(unittest.TestCase):
    def test_neuron(self):
        self.assertAlmostEqual(neuron([2, 3, 4], [0.5, -1, 2], 1), 7)
        with self.assertRaises(ValueError):
            neuron([2, 3, 4], [0.5, -1], 1)

    def test_relu(self):
        for z, value, derivative in [(-5, 0, 0), (0, 0, 0), (3, 3, 1)]:
            self.assertEqual(relu(z), value)
            self.assertEqual(relu_derivative(z), derivative)

    def test_layer(self):
        hidden = layer([2, 3], [[0.5, -1], [1, 0.5], [-0.5, 2]], [1, 0, -1])
        self.assertEqual(hidden, [0, 3.5, 4])
        self.assertAlmostEqual(layer(hidden, [[0.5, 0.3, 0.2]], [0])[0], 1.85)
        with self.assertRaises(ValueError):
            layer([1], [[1]], [])

    def test_loss(self):
        self.assertAlmostEqual(squared_error(1.85, 3), 1.3225)
        self.assertEqual(squared_error_derivative(24, 20), 8)
        self.assertEqual(mean_squared_error([2, 5, 3, 8], [3, 4, 5, 6]), 2.5)
        with self.assertRaises(ValueError):
            mean_squared_error([], [])
        with self.assertRaises(ValueError):
            mean_squared_error([1], [1, 2])

    def test_update(self):
        self.assertAlmostEqual(update_weight(3, 64, 0.01), 2.36)
        self.assertAlmostEqual(update_weight(3, -64, 0.01), 3.64)

    def test_gradients(self):
        self.assertEqual(gradients(2, 20, 3, 4), (64, 48))
        epsilon = 1e-5
        def loss(w1, w2):
            return squared_error(forward(2, w1, w2)[1], 20)
        numeric_w1 = (loss(3 + epsilon, 4) - loss(3 - epsilon, 4)) / (2 * epsilon)
        numeric_w2 = (loss(3, 4 + epsilon) - loss(3, 4 - epsilon)) / (2 * epsilon)
        self.assertAlmostEqual(numeric_w1, 64, places=6)
        self.assertAlmostEqual(numeric_w2, 48, places=6)

    def test_training(self):
        data = [(x, x * 10) for x in range(1, 6)]
        w1, w2, history = train(data, 3, 4)
        self.assertEqual(len(history), 100)
        self.assertLess(history[-1], history[0])
        self.assertAlmostEqual(forward(10, w1, w2)[1], 100, places=5)


if __name__ == "__main__":
    unittest.main()
