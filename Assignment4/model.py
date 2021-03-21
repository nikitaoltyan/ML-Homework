import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net
    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network
        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.input_shape = input_shape

        width, height, n_channels = input_shape
        
        # computing FCL input. All numbers are taken
        # from class description (except padding. I took 1).
        pool_stride, filter_size = 4, 4
        padding = 1
        kernel_size = 3

        self.fcl_input = int(height/4)
        self.n_output_classes = n_output_classes

        self.layer1 = ConvolutionalLayer(n_channels, conv1_channels, kernel_size, padding)
        self.layer2 = ReLULayer()
        self.layer3 = MaxPoolingLayer(filter_size, pool_stride)
        self.layer4 = ConvolutionalLayer(conv1_channels, conv2_channels, kernel_size, padding)
        self.layer5 = ReLULayer()
        self.layer6 = MaxPoolingLayer(filter_size, pool_stride)
        self.layer7 = Flattener()
        self.layer8 = FullyConnectedLayer(self.fcl_input, n_output_classes)


    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment

        params = self.params()
        for par in params:
          param = params[par]
          param.grad = np.zeros_like(param.grad)
        
        fw1 = self.layer1.forward(X)
        fw2 = self.layer2.forward(fw1)
        fw3 = self.layer3.forward(fw2)
        fw4 = self.layer4.forward(fw3)
        fw5 = self.layer5.forward(fw4)
        fw6 = self.layer6.forward(fw5)
        fw7 = self.layer7.forward(fw6)
        fw8 = self.layer8.forward(fw7)

        loss, pred = softmax_with_cross_entropy(fw8, y)
        
        bw1 = self.layer8.backward(pred)
        bw2 = self.layer7.backward(bw1)
        bw3 = self.layer6.backward(bw2)
        bw4 = self.layer5.backward(bw3)
        bw5 = self.layer4.backward(bw4)
        bw6 = self.layer3.backward(bw5)
        bw7 = self.layer2.backward(bw6)
        bw8 = self.layer1.backward(bw7)

        #for par in params:
        #  param = params[par]
        #  param.grad += l2_regularization(param.value, self.reg)[1]

        return loss


    def predict(self, X):
        # You can probably copy the code from previous assignment
        raise Exception("Not implemented!")

    def params(self):
        # TODO: Aggregate all the params from all the layers
        # which have parameters
        return {'W1': self.layer1.W, 'B1': self.layer1.B, 
                'W4': self.layer4.W, 'B4': self.layer4.B,
                'W8': self.layer8.W, 'B8': self.layer8.B}
