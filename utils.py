import numpy as np
import pandas as pd


def leaky_relu(x):
    # return np.maximum(x, 0)
    return np.where(x > 0, x, 0.01 * x)



def deriv_leaky_relu(x):
    # return np.where(x > 0, 1, 0)
    return np.where(x > 0, 1, 0.01)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def bce_loss(yTrue, yPred):
    epsilon = 1e-10
    yPred = np.clip(yPred, epsilon, 1 - epsilon)
    return -(yTrue * np.log(yPred) + (1-yTrue) * np.log(1-yPred))


def deriv_bce(yTrue, yPred):
    epsilon = 1e-10
    yPred = np.clip(yPred, epsilon, 1 - epsilon)
    return -(yTrue/yPred - (1-yTrue) / (1-yPred))


def normalize_features(x):
    epsilon = 1e-5
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + epsilon)


def clip_gradients(gradient):
    return np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)