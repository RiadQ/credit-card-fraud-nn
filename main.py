from dataset import load_data, get_positive_samples, create_mini_batches
from model import NeuralNetwork
import numpy as np
from random import randint


if __name__ == '__main__':
    x, y = load_data('creditcard.csv', 5)
    batches = create_mini_batches(x, y, 32)

    model = NeuralNetwork(batches)

    true_pred = np.empty(0)
    false_pred = np.empty(0)

    model.train()

    true_features = get_positive_samples(x, y)
    for i in range(400):
        false = model.compute(x[randint(0, 0.28e5)])
        true = model.compute(true_features[i])
        false_pred = np.append(false_pred, false)
        true_pred = np.append(true_pred, true)

    print(f'Mean value true: {true_pred.mean()}')
    print(f'Mean value false: {false_pred.mean()}')
