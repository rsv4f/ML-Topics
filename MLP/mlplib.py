from mlp_functions import *


# taken ideas from
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/neural_network/_multilayer_perceptron.py
# https://www.youtube.com/watch?v=Ilg3gGewQ5U

class MultiLayerPerception:
    """
    Implements multi-layer perceptron
    """

    def __init__(self, dims=[1, 2, 4, 1], eta=0.001, activation='sigmoid',
                 max_epochs=10000, deltaE=-np.inf, alpha=0.8):
        """
        dims = [dim_in, dim_hidden1, ..., dim_hiddenN, dim_out]
        eta = leraning rate (used in gradient decent)
        activation = activation function (relu, sigmoid)
        max_epochs = maximum number of epochs during training
        deltaE = stopping criterion
        alpha = momentum parameter
        """
        self.set_params(dims, eta, activation, max_epochs, deltaE, alpha)

    # compatibility with sklearn 
    def set_params(self, dims, eta, activation, max_epochs, deltaE, alpha):
        self.dims = dims
        self.eta = eta
        self.activation = activation
        self.max_epochs = max_epochs
        self.deltaE = deltaE
        self.alpha = alpha
        self.dW = []  # momentum terms
        if activation == 'sigmoid':
            self.f = SIG
            self.df = dSIG
        elif activation == 'relu':
            self.f = ReLU
            self.df = dReLU
        else:
            raise ValueError(f"invalid activation function {activation}")

        return self

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("training and target shapes don't match")
        # initialize weights
        self.weights = []
        for i in range(len(self.dims) - 1):
            W = np.random.rand(self.dims[i + 1], self.dims[i] + 1) - 0.5
            #                  ^ output dim    ^ input dim plus bias dim
            # W = (W.T/np.sum(W, axis=1)).T # normalize ROWS for mid-range output
            self.weights.append(W)
        # initial momentum terms 
        for W in self.weights:
            self.dW.append(np.zeros(W.shape))
        # store error values 
        self.train_error = np.zeros(self.max_epochs + 1)
        self.test_error = np.zeros(self.max_epochs + 1)
        self.train_error[-1] = np.infty
        self.test_error[-1] = np.infty
        # main training loop
        t = 0
        while t < self.max_epochs:
            # shuffle data
            Xs = X
            ys = y
            # forward pass 
            Y, x, u = self._forwardpass(Xs, self.weights)
            # compute error 
            rmse = self._RMSE(Y, ys)
            self.train_error[t] = rmse
            delta = self.train_error[t] - self.train_error[t - 1]
            if abs(delta) < self.deltaE:
                break
            else:
                # backward pass
                self.weights = self._backwardpass(ys, Y, x, u, self.weights)
                t = t + 1
        self.train_error = self.train_error[:t]
        self.test_error = self.test_error[:t]
        # self.weights = weights
        return self.weights

    def predict(self, X):
        Y, nothing, nothing2 = self._forwardpass(X, self.weights)
        return Y

    def _forwardpass(self, X, weights):
        """ perform forward pass, saving values"""
        Y = X
        x = []  # inputs to next layer
        u = []  # activations
        for i in range(len(weights)):
            X = addOnesCol(Y)
            x.append(X)  # save input
            U = (weights[i] @ X.T).T  # apply weight matrix
            u.append(U)  # save output
            Y = self.f(U)  # activated output
        return Y, x, u

    def _RMSE(self,y, yp):
        """
        Computes the root mean squared error (RMSE)
        """
        return np.sqrt(np.sum((yp-y)**2)/y.shape[0])

    def _backwardpass(self, y, Y, x, u, weights):
        """ 
        Compute updated weights by doing backward pass
        y = target 
        Y = true output 
        x = inputs to weight matrices at each layer during forward pass
        u = activations at each output layer during forward pass 
        """
        # backward pass
        D = -self.df(u[-1]) * (y - Y)  # Delta
        delta = [D]
        for i in range(len(weights) - 1):
            W = weights[::-1][i]  # go through weight matrices in reverse
            U = u[::-1][i + 1]  # go through outputs in reverse, from second last
            d = self.df(U) * (delta[i] @ W)[:, 1:]
            delta.append(d)
        delta.reverse()  # reverse delta!
        # update weights 
        weights_new = []
        for i in range(len(weights)):
            W = weights[i]
            momentum = self.alpha * self.dW[i]
            learningTerm = self.eta * (delta[i].T @ x[i])
            Wnew = W - learningTerm + momentum
            self.dW[i] = Wnew - W
            weights_new.append(Wnew)
        return weights_new
