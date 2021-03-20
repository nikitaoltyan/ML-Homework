import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    z = predictions.copy()
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
        
class ReLULayer:
    def __init__(self):
        # I added it insted of cache I used in previous assignment. I think it's easier to use.
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        X = np.maximum(X, 0)
        return X

    def backward(self, d_out):
        # TODO copy from the previous assignment
        X = self.X
        mask = X > 0
        d_in =  mask * d_out
        return d_in

    def params(self):
        return {}


    
class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        Xw = X.dot(self.W.value)
        out = np.add(Xw, self.B.value)
        return out

    def backward(self, d_out):
        # TODO copy from the previous assignment
        xT = self.X.T
        ones = np.ones((xT.shape[1], 1)).T
        dB = np.dot(ones, d_out)

        self.B.grad += dB   
        self.W.grad += xT.dot(d_out) 

        d_input = d_out.dot(self.W.value.T)
        return d_input

    
    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.padding = padding
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        pool_size = self.pool_size
        stride = self.stride
        out_height = 1 + (height - pool_size) // stride
        out_width = 1 + (width - pool_size) // stride
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                height_start = y * stride
                height_end = height_start + pool_size
                width_start = x * stride
                width_end = width_start + pool_size
                X_slice = X[:, height_start:height_end, 
                                      width_start:width_end, :]
                output[:, y, x, :] = np.max(X_slice, axis=(1, 2))
                
        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        X = self.X
        padding = self.padding
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        d_input = np.zeros_like(X)
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        W = self.W.value.reshape((-1, out_channels))



        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        filter_size = self.filter_size
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                point = d_out[:, y, x, :]
                
                X_after_kernel = X[:, y:y+filter_size, x:x+filter_size, :]
                X_after_kernel_reshaped = X_after_kernel.reshape((batch_size, -1))
                X_t = X_after_kernel_reshaped.T

                d_W = X_t.dot(point)
                d_W = d_W.reshape((filter_size, filter_size, self.in_channels, out_channels))
                
                ones_t = np.ones((batch_size, )).T
                d_B = ones_t.dot(point)
                
                d_X_before_kernel = point.dot(W.T)
                d_X_before_kernel = d_X_before_kernel.reshape((batch_size, filter_size, filter_size, self.in_channels))
                
                self.W.grad += d_W
                self.B.grad += d_B
                d_input[:, y:y+filter_size, x:x+filter_size, :] += d_X_before_kernel

        # Delete padding if exists.
        d_input = d_input[:, padding:height-padding, padding:width-padding, :]
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool
        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = int(height/2)
        out_width = int(width/2)
        result = np.zeros((batch_size, out_height, out_width, channels))
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(channels):
                        y_source = y * self.stride
                        x_source = x * self.stride
                        pool = X[batch, y_source:y_source+self.pool_size, x_source:x_source+self.pool_size, channel]
                        try:
                          maximum = np.max(pool)
                        except ValueError:  #raised if `pool` is empty.
                          pass
                        # maximum = np.max(pool)
                        result[batch, y, x, channel] = maximum
                #result[:, y, x, :] = Xk.dot(W) + self.B.value
        return result
        

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        pool_size = self.pool_size
        stride = self.stride
        X = self.X
        batch_size, height, width, in_channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros_like(X)
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(in_channels):
                        mask = np.zeros((pool_size, pool_size))
                        y_source = y * stride
                        x_source = x * stride
                        pool = X[batch,
                                 y_source:np.minimum(y_source+pool_size, height),
                                 x_source:np.minimum(x_source+pool_size, width), channel]
                        
                        try:
                          maximum = np.max(pool)
                        except ValueError:  #raised if `pool` is empty.
                          pass
                        # maximum = np.max(pool)
                        max_count = np.count_nonzero(pool == maximum)
                        argmax = np.argwhere(pool==maximum)
                        mask[argmax[:,0], argmax[:,1]] = d_out[batch, y, x, channel] / max_count
                        
                        d_input[batch, y_source:y_source+pool_size, x_source:x_source+pool_size, channel] += mask
                        
        return d_input

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X_shape = X.shape
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, -1))

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X_shape
        
        return d_out.reshape((batch_size, height, width, channels))

    def params(self):
        # No params!
        return {}
