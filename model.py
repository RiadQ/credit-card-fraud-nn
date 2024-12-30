import numpy as np

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bce_loss(yTrue, yPred):
    return -(yTrue * np.log(yPred) + (1-yTrue) * np.log(1-yPred))


class hiddenLayer:
    def __init__(self, matrix):
        self.W = matrix
        self.b = np.array([0.72, -0.34, 0.59, -0.77, 0.44, -0.61, 0.89, -0.12, 0.53, -0.25])

    def feedforward(self, x):
        return relu(self.W @ x + self.b)


class outputLayer:
    def __init__(self):
        self.W = np.array([[0.72, -0.12, 0.34, -0.95, 0.64, -0.45, 0.85, -0.32, 0.76, 0.24]])
        self.b = 0.2

    def feedforward(self, x):
        return sigmoid(self.W @ x + self.b)


class NeuralNetwork:
    def __init__(self):
        self.h1 = hiddenLayer(np.array([
            [0.72, -0.12, 0.34, -0.95, 0.64, -0.45, 0.85, -0.32, 0.76, 0.24, -0.18, 0.44, 0.67, -0.91, 0.09, 0.51, -0.77, 0.58, 0.36, -0.15, 0.92, 0.81, -0.59, 0.39, -0.67, 0.23, 0.88, -0.04, 0.49],
            [-0.48, 0.96, -0.71, 0.18, 0.33, -0.81, 0.57, -0.65, 0.45, 0.26, -0.38, 0.87, 0.29, -0.14, 0.61, -0.27, 0.83, -0.36, 0.97, -0.74, 0.12, 0.49, -0.56, 0.21, -0.89, 0.73, -0.31, 0.62, -0.41],
            [0.25, -0.98, 0.41, -0.66, 0.79, -0.58, 0.93, -0.22, 0.54, -0.35, 0.46, -0.84, 0.76, 0.19, -0.44, 0.63, -0.37, 0.12, -0.61, 0.91, -0.39, 0.77, -0.42, 0.18, 0.53, -0.26, 0.87, -0.49, 0.31],
            [0.61, -0.87, 0.29, -0.64, 0.81, -0.49, 0.96, -0.19, 0.73, -0.52, 0.35, -0.78, 0.69, 0.11, -0.56, 0.48, -0.43, 0.14, -0.75, 0.84, -0.25, 0.82, -0.57, 0.17, 0.47, -0.34, 0.91, -0.46, 0.22],
            [-0.77, 0.38, -0.12, 0.61, -0.29, 0.87, -0.54, 0.19, -0.33, 0.64, -0.22, 0.41, -0.81, 0.28, 0.52, -0.73, 0.46, 0.18, -0.79, 0.33, 0.65, -0.48, 0.21, -0.62, 0.39, -0.35, 0.83, -0.96, 0.24],
            [0.87, -0.53, 0.14, -0.76, 0.48, -0.39, 0.92, -0.65, 0.23, -0.44, 0.71, -0.19, 0.85, -0.28, 0.59, -0.31, 0.77, -0.22, 0.46, -0.58, 0.91, -0.24, 0.37, -0.63, 0.29, -0.57, 0.82, -0.49, 0.45],
            [-0.36, 0.89, -0.64, 0.28, -0.45, 0.73, -0.26, 0.47, -0.54, 0.91, -0.31, 0.85, -0.29, 0.67, -0.21, 0.41, -0.86, 0.63, -0.38, 0.76, -0.55, 0.28, -0.72, 0.39, -0.41, 0.94, -0.66, 0.24, -0.18],
            [0.68, -0.79, 0.24, -0.55, 0.93, -0.21, 0.49, -0.84, 0.37, -0.26, 0.65, -0.11, 0.72, -0.34, 0.57, -0.61, 0.89, -0.39, 0.22, -0.48, 0.64, -0.35, 0.71, -0.19, 0.95, -0.41, 0.88, -0.25, 0.77],
            [-0.73, 0.49, -0.16, 0.67, -0.42, 0.85, -0.24, 0.71, -0.58, 0.33, -0.91, 0.46, -0.29, 0.92, -0.61, 0.48, -0.35, 0.87, -0.49, 0.74, -0.26, 0.89, -0.12, 0.38, -0.52, 0.96, -0.77, 0.41, -0.65],
            [0.39, -0.83, 0.21, -0.56, 0.72, -0.47, 0.64, -0.35, 0.91, -0.18, 0.87, -0.44, 0.29, -0.67, 0.58, -0.22, 0.76, -0.51, 0.95, -0.43, 0.68, -0.39, 0.23, -0.57, 0.82, -0.26, 0.74, -0.48, 0.62]
        ]))
        self.h2 = hiddenLayer(np.array([
            [0.72, -0.12, 0.34, -0.95, 0.64, -0.45, 0.85,