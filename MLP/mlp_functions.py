import numpy as np

def SIG(x):
    """ sigmoid function """
    return 1 / (1 + np.exp(-x))


def dSIG(x):
    """ derivative of sigmoid """
    return np.exp(-x) / (1 + np.exp(-x)) ** 2


def ReLU(x):
    """ rectifier function """
    result = x * (x > 0)
    return result


def dReLU(x):
    """ derivative of rectifier """
    return 1. * (x > 0)

def addOnesCol(X):
    """ add column of ones """
    a, b = X.shape
    Xt = np.zeros([a, b + 1])
    Xt[:, 1:] = X
    Xt[:, 0] = np.ones(a)
    return Xt