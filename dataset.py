import numpy as np
import pandas


def load_data(filepath):
    data = pandas.read_csv(filepath)
    feature_vectors = data.iloc[:, 1:-1].values.astype(np.float32)
    true_labels = data.iloc[:, -1].values.astype(np.float32)
    return feature_vectors, true_labels


def get_positive_samples(features, lables):
    positive_indicies = np.where(lables == 1)[0]
    return true_features[positive_indicies]