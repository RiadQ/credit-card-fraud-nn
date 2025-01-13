import numpy as np
import pandas as pd

def load_data(filepath):
    data = pd.read_csv(filepath)
    feature_vectors = data.iloc[:, 1:-1].values.astype(np.float32)
    true_labels = data.iloc[:, -1].values.astype(np.float32)
    return feature_vectors, true_labels

def create_mini_batches(X, y, batch_size):
    n_samples = X.shape[0]
    indicies = np.arange(n_samples)
    np.random.shuffle(indicies)

    x_shuffled = X[indicies]
    y_shuffled = y[indicies]

    batches = []
    for i in range(0, n_samples, batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batches.append((x_batch, y_batch))

    return batches


def get_positive_samples(features, lables):
    positive_indicies = np.where(lables == 1)[0]
    return true_features[positive_indicies]


X, y = load_data('creditcard.csv')

data = create_mini_batches(X, y, 32)

# print(np.mean(np.stack(data[0][0]), axis=0), data[0][1])

print(data[0])

