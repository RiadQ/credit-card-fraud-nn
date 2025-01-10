import numpy as np
from utils import leaky_relu, deriv_leaky_relu, bce_loss, deriv_bce, sigmoid, clip_gradients


class hiddenLayer:
    def __init__(self, inputs, neurons):
        self.W = np.random.randn(neurons, inputs).astype(np.float32) * np.sqrt(2 / inputs)
        self.b = np.zeros(neurons,).astype(np.float32)

    def feedforward(self, x):
        return leaky_relu(np.dot(self.W, x) + self.b)


class outputLayer:
    def __init__(self):
        self.W = np.random.randn(1, 10).astype(np.float32) * 0.1
        self.b = 0

    def feedforward(self, x):
        return sigmoid(np.dot(self.W, x) + self.b)


class NeuralNetwork:
    def __init__(self, features, lables):
        self.h1 = hiddenLayer(29, 10)
        self.h2 = hiddenLayer(10, 10)
        self.o1 = outputLayer()
        self.epochs = 10
        self.learn_rate = 0.01
        self.feature_vectors = features
        self.yTrues = lables

    def compute(self, x):
        self.h1_out = self.h1.feedforward(x)
        self.h2_out = self.h2.feedforward(self.h1_out)
        return self.o1.feedforward(self.h2_out)

    def train(self):
        for epoch in range(self.epochs):
            loop = 1
            for i, vector in enumerate(self.feature_vectors):
                vector = normalize_features(vector)
                yPred = self.compute(vector)
                loss = bce_loss(yPred, self.yTrues[i])
                
                dL_dy = clip_gradients(deriv_bce(self.yTrues[i], yPred))                         
                dy_do1 = clip_gradients(yPred * (1 - yPred))
                do1_dx3 = self.o1.W.T
                dx3_h2 = deriv_leaky_relu(np.dot(self.h2.W, self.h1_out) + self.h2.b)
                dh2_dx2 = self.h2.W.T
                dx2_dh1 = deriv_leaky_relu(np.dot(self.h1.W, vector) + self.h1.b)

                dL_do1 = clip_gradients(dL_dy * dy_do1)
                dL_dx3 = clip_gradients(dL_do1 * do1_dx3.T)
                dL_dh2 = clip_gradients(dL_dx3 @ dx3_h2)
                dL_dx2 = clip_gradients(dL_dh2 * dh2_dx2.T)
                dL_dh1 = clip_gradients(dL_dx2 @ dx2_dh1)

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

                print(f'Epoch: {epoch + 1} Example: {loop}')
                loop += 1
