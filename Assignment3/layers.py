import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    z = predictions.copy()
    if predictions.ndim == 1:
        z = z.reshape(1, -1)
    z -= np.max(z, axis=1).reshape(-1, 1)
    exps = np.exp(z)
    sums = np.sum(exps, axis=1)
    probs = exps / sums.reshape(-1, 1)

    return probs

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # Copy from the previous assignment
    z = preds.copy()
    N = len(z)
    z -= np.max(z, axis=1).reshape(-1, 1)
    exps = np.exp(z)
    sums = np.sum(exps, axis=1)
    probs = exps / sums.reshape(-1, 1)

    p_pred = np.zeros(N)
    for n in range(N):
        p_pred[n] = probs[n, target_index[n]]
    # print(N)
    loss = - 1 / N * np.sum(np.log(p_pred))

    d_preds = probs.copy()
    for n in range(N):
        d_preds[n, target_index[n]] -= 1
    d_preds /= N
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        cache = X
        X = np.maximum(X, 0)
        return X, cache

    def backward(self, d_out, cashe):
        """
        Backward pass
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        # raise Exception("Not implemented!")

        X = cashe

        mask = X > 0
        d_in =  mask * d_out

        return d_in

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
   def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

   def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        cache = X
        Xw = X.dot(self.W.value)
        out = np.add(Xw, self.B.value)

        return out, cache


   def backward(self, d_out, cache):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B
        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output
        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")

        xT = cache.T
        ones = np.ones((xT.shape[1], 1)).T

        dB = np.dot(ones, d_out)

        self.B.grad += dB   
        self.W.grad += xT.dot(d_out) 

        d_input = d_out.dot(self.W.value.T)

        return d_input

   def params(self):
        return {'W': self.W, 'B': self.B}
