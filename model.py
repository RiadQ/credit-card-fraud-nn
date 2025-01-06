import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def bce_loss(yTrue, yPred):
    epsilon = 1e-10
    yPred = np.clip(yPred, epsilon, 1 - epsilon)
    return -(yTrue * np.log(yPred) + (1-yTrue) * np.log(1-yPred))


def deriv_bce(yTrue, yPred):
    epsilon = 1e-10
    yPred = np.clip(yPred, epsilon, 1 - epsilon)
    return -(yTrue/yPred - (1-yTrue) / (1-yPred))


def deriv_relu(x):
    return (0 < x).astype(float)


def normalize_features(x):
    epsilon = 1e-5
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + epsilon)

def clip_gradients(gradient, max_value=1.0):
    return np.clip(gradient, -max_value, max_value)


def clip_gradients(gradient, max_value=1.0):
    # Add value checking to avoid NaN propagation
    if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
        return np.zeros_like(gradient)
    return np.clip(gradient, -max_value, max_value)


data = pd.read_csv('creditcard.csv')

feature_vectors = data.iloc[:, 1:-1].values.astype(np.float32)

true_labels = data.iloc[:, -1].values.astype(np.float32)

num_columns = feature_vectors.shape[1]

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
    nearest = sorted_pairs[:k]
    enumerated = list(enumerate(calculate_nearest_neighbour(vector)))
    sorted_pairs = sorted(enumerated, key=lambda x: x[1])
    l = np.random.uniform(0, 1)
    for d in nearest:
        global feature_vectors, true_labels
        n = true_features[d[0]]
        synthetic = vector + l * (n - vector)
        feature_vectors = np.vstack([feature_vectors, synthetic])
        true_labels = np.append(true_labels, 1)


class hiddenLayer:
    def __init__(self, inputs, neurons):
        self.W = np.random.randn(neurons, inputs).astype(np.float32) * np.sqrt(1 / 11) * 0.1
        self.b = np.random.randn(neurons,).astype(np.float32)

    def feedforward(self, x):
        return relu(np.dot(self.W, x) + self.b)

class outputLayer:
    def __init__(self):
        self.W = np.random.randn(1, 10).astype(np.float32) * np.sqrt(1 / 11) * 0.1
        self.b = 0

    def feedforward(self, x):
        return sigmoid(np.dot(self.W, x) + self.b)


class NeuralNetwork:
    def __init__(self):
        self.h1 = hiddenLayer(29, 10)
        self.h2 = hiddenLayer(10, 10)
        self.o1 = outputLayer()
        self.epochs = 10
        self.learn_rate = 0.001

    def compute(self, x):
        self.h1_out = self.h1.feedforward(x)
        self.h2_out = self.h2.feedforward(self.h1_out)
        return self.o1.feedforward(self.h2_out)

    def train(self, x, yTrues):
        for epoch in range(self.epochs):
            loop = 1
            for i, vector in enumerate(x):
                vector = normalize_features(vector)
                yPred = self.compute(vector)
                loss = bce_loss(yPred, yTrues[i])
                
                dL_dy = clip_gradients(deriv_bce(yTrues[i], yPred))                               
                dy_do1 = clip_gradients(yPred * (1 - yPred))
                do1_dx3 = self.o1.W.T
                dx3_h2 = deriv_relu(np.dot(self.h2.W, self.h1_out) + self.h2.b)
                dh2_dx2 = self.h2.W.T
                dx2_dh1 = deriv_relu(np.dot(self.h1.W, vector) + self.h1.b)

                scale_factor = 1.5
                dL_do1 = clip_gradients(dL_dy * dy_do1 * scale_factor)
                dL_dx3 = clip_gradients(dL_do1 * do1_dx3.T * scale_factor)
                dL_dh2 = clip_gradients(dL_dx3 @ dx3_h2 * scale_factor)
                dL_dx2 = clip_gradients(dL_dh2 * dh2_dx2.T * scale_factor)
                dL_dh1 = clip_gradients(dL_dx2 @ dx2_dh1 * scale_factor)

                dL_dw1 = np.outer(dL_dh1, vector)
                dL_dw2 = np.outer(dL_dh2, self.h1_out)
                dL_dw3 = np.outer(dL_do1, self.h2_out)

                dL_db1 = dL_dh1
                dL_db2 = dL_dh2
                dL_db3 = dL_do1

                self.h1.W -= self.learn_rate * dL_dw1
                self.h2.W -= self.learn_rate * dL_dw2
                self.o1.W -= self.learn_rate * dL_dw3

                self.h1.b -= self.learn_rate * dL_db1
                self.h2.b -= self.learn_rate * dL_db2
                self.o1.b -= self.learn_rate * dL_db3


                print(loop)
                loop += 1


        print(nn.compute(feature_vectors[3]))
        print(true_labels[3])
        print(nn.compute(feature_vectors[257200]))
        print(true_labels[257200])
        print(nn.compute(feature_vectors[100000]))
        print(true_labels[100000])
        print(nn.compute(feature_vectors[623]))
        print(true_labels[623])


nn = NeuralNetwork()

nn.train(feature_vectors, true_labels)
