import numpy as np
import pandas as pd

def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bce_loss(yTrue, yPred):
    return -(yTrue * np.log(yPred) + (1-yTrue) * np.log(1-yPred))


def deriv_bce(yTrue, yPred):
    return -((yTrue/yPred) + ((1-yTrue) / 1-yPred))


def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def deriv_relu(x):
    if x > 0:
        return 1
    else:
        return 0

# Normalizes the 'Amount' column to values between 0 and 1. This is done to avoid exploding activation values
# def min_max_norm(column):
#     return (column - np.min(column)) / (np.max(column) - np.min(column))


data = pd.read_csv('creditcard.csv')

# data['Amount'] = min_max_norm(data.iloc[:, -2])

feature_vectors = data.iloc[:, 1:-1].values

true_labels = data.iloc[:, -1].values


num_columns = feature_vectors.shape[1]

# Normalize each column using Z-Score Normalization
for col in range(num_columns):
    column_mean = np.mean(feature_vectors[:, col])
    column_std = np.std(feature_vectors[:, col])
    if column_std != 0:  # Avoid division by zero
        feature_vectors[:, col] = (feature_vectors[:, col] - column_mean) / column_std



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
        # Weight matrices are initialized manually for learning purposes. I might change this later
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
            [0.72, -0.12, 0.34, -0.95, 0.64, -0.45, 0.85, -0.32, 0.76, 0.76],
            [-0.48, 0.96, -0.71, 0.18, 0.33, -0.81, 0.57, -0.65, 0.45, 0.76],
            [0.25, -0.98, 0.41, -0.66, 0.79, -0.58, 0.93, -0.22, 0.54, 0.76],
            [0.61, -0.87, 0.29, -0.64, 0.81, -0.49, 0.96, -0.19, 0.73, 0.76],
            [-0.77, 0.38, -0.12, 0.61, -0.29, 0.87, -0.54, 0.19, -0.33, 0.76],
            [0.87, -0.53, 0.14, -0.76, 0.48, -0.39, 0.92, -0.65, 0.23, 0.76],
            [-0.36, 0.89, -0.64, 0.28, -0.45, 0.73, -0.26, 0.47, -0.54, 0.76],
            [0.68, -0.79, 0.24, -0.55, 0.93, -0.21, 0.49, -0.84, 0.37, 0.76],
            [-0.73, 0.49, -0.16, 0.67, -0.42, 0.85, -0.24, 0.71, -0.58, 0.76],
            [0.39, -0.83, 0.21, -0.56, 0.72, -0.47, 0.64, -0.35, 0.91, 0.76]
        ]))
        self.o1 = outputLayer()
        self.epochs = 10

    def compute(self, x):
        h1_out = self.h1.feedforward(x)
        h2_out = self.h2.feedforward(h1_out)
        return self.o1.feedforward(h2_out)

    def train(self, x, yTrues):
        for i in range(self.epochs):
            pass
        loss = bce_loss(self.compute(x), yTrue)


nn = NeuralNetwork()


x = np.array([-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,0.1])
x = x.reshape(29,)


print(nn.compute(x))
row = feature_vectors[13973]
print('Row values:\n', row)
print('Length: ', len(row))
print(nn.compute(row))
