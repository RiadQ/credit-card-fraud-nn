from dataset import load_data, get_positive_samples
from model import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    feature_vectors, true_labels = load_data('creditcard.csv')
    true_features = get_positive_samples(feature_vectors, true_labels)

    model = NeuralNetwork(feature_vectors, true_labels)

    true_pred = np.empty(0)
    false_pred = np.empty(0)

    for i in range(100, 200):
        false = nn.compute(feature_vectors[i])
        true = nn.compute(true_features[i])
        print(f'True value: {true_labels[i]} Prediction: {false}')
        print(f'True value: {true_labels[positive_indicies[i]]} Prediction: {true}')
        false_pred = np.append(false_pred, false)
        true_pred = np.append(true_pred, true)


    print(f'Mean value true: {true_pred.mean()}')
    print(f'Mean value false: {false_pred.mean()}')
