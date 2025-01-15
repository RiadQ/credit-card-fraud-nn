import numpy as np
import pandas as pd

data = pd.read_csv('creditcard.csv')

feature_vectors = data.iloc[:, 1:-1].values.astype(np.float32)

true_labels = data.iloc[:, -1].values.astype(np.float32)

positive_indicies = np.where(true_labels == 1)[0]

true_features = feature_vectors[positive_indicies]

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


def synthetic_sample(vector, k=5):
    enumerated = list(enumerate(calculate_nearest_neighbour(vector)))
    sorted_pairs = sorted(enumerated, key=lambda x: x[1])
    nearest = sorted_pairs[:k]
    l = np.random.uniform(0, 1)
    for d in nearest:
        global feature_vectors, true_labels
        n = true_features[d[0]]
        synthetic = vector + l * (n - vector)

        feature_vectors = np.vstack([feature_vectors, synthetic])
        true_labels = np.append(true_labels, 1)