import numpy as np
import pandas as pd
from smote import synthetic_sample, positive_indicies


def calculate_nearest_neighbour(yTrue_vector):
    distances = []
    global true_features
    for vector in true_features:
        sums = 0
        for i, value in enumerate(yTrue_vector):
            sums += (value - vector[i])**2
        
        sums = np.sqrt(sums)
        distances.append(sums)

    return distances


def synthetic_sample(vector, feature_vectors, k=5):
    enumerated = list(enumerate(calculate_nearest_neighbour(vector)))
    sorted_pairs = sorted(enumerated, key=lambda x: x[1])
    nearest = sorted_pairs[:k]
    l = np.random.uniform(0, 1)
    for d in nearest:
        global true_labels
        n = true_features[d[0]]
        synthetic = vector + l * (n - vector)

        feature_vectors = np.vstack([feature_vectors, synthetic])
        true_labels = np.append(true_labels, 1)




def load_data(filepath, synthetic_samples=None):
    data = pd.read_csv(filepath)
    feature_vectors = data.iloc[:, 1:-1].values.astype(np.float32)
    true_labels = data.iloc[:, -1].values.astype(np.float32)

    positive_indicies = np.where(true_labels == 1)[0]
    true_features = feature_vectors[positive_indicies]

    if synthetic_samples:
        for vector in positive_indicies[:synthetic_samples]:
            
            distances = []
            for vector in true_features:
                sums = 0
                for i, value in enumerate(vector):
                    sums += (value - vector[i])**2
                    
                    sums = np.sqrt(sums)
                    distances.append(sums)
            
            enumerated = list(enumerate(distances))
            sorted_pairs = sorted(enumerated, key=lambda x: x[1])
            nearest = sorted_pairs[:1]
            l = np.random.uniform(0, 1)

            for d in nearest:
                n = true_features[d[0]]
                synthetic = vector + l * (n - vector)

                feature_vectors = np.vstack([feature_vectors, synthetic])
                true_labels = np.append(true_labels, 1)

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
    return features[positive_indicies]

