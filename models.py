import numpy as np
import random


def l2_loss(predictions,Y):
    '''
        Computes L2 loss (sum squared loss) between true values, Y, and predictions.

        :param Y: A 1D Numpy array with real values (float64)
        :param predictions: A 1D Numpy array of the same size of Y
        :return: L2 loss using predictions for Y.
    '''
    # TODO
    difference_array = np.subtract (Y, predictions)
    return np.dot (difference_array, difference_array)

    pass

def sigmoid(x):
    '''
        Sigmoid function f(x) =  1/(1 + exp(-x))
        :param x: A scalar or Numpy array
        :return: Sigmoid function evaluated at x (applied element-wise if it is an array)
    '''
    return np.where(x > 0, 1 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def sigmoid_derivative(x):
    '''
        First derivative of the sigmoid function with respect to x.
        :param x: A scalar or Numpy array
        :return: Derivative of sigmoid evaluated at x (applied element-wise if it is an array)
    '''
    # TODO
    if np.isscalar (x):
        return (sigmoid(x) * (1 - sigmoid(x)))
    else:
        num_examples = x.size
        derivatives_array = np.zeros(num_examples)
        for i in range (num_examples):
            derivatives_array[i] = (sigmoid(x[i]) * (1 - sigmoid(x[i])))

    return derivatives_array

class OneLayerNN:
    '''
        One layer neural network trained with Stocastic Gradient Descent (SGD)
    '''
    def __init__(self):
        '''
        @attrs:
            weights The weights of the neural network model.
        '''
        self.weights = None

    def train(self, X, Y, learning_rate=0.001, epochs=250, print_loss=True):
        '''
        Trains the OneLayerNN model using SGD.

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param learning_rate: The learning rate to use for SGD
        :param epochs: The number of times to pass through the dataset
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # Initializing the weights to random values
        num_attributes = X.shape[1]
        num_examples = X.shape[0]
        self.weights = np.zeros(num_attributes)
        derivatives_array = np.zeros(num_attributes)

        # Running gradient descent
        for i in range (epochs):

            for j in range (num_examples):
                #Calculating the Delta for the last node
                output = self.predict(X[j])
                delta = 2 * (Y[j] - output)

                # Calculating the derivative for each weight
                for k in range (num_attributes):
                    derivatives_array[k] = derivatives_array[k] + (-X[j][k] * delta)

                self.weights = self.weights - (learning_rate * derivatives_array)
                derivatives_array = np.zeros(num_attributes)


            if (print_loss):
                print (self.average_loss(X, Y))

        pass

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X: 2D Numpy array where each row contains an example.
        :return: A 1D Numpy array containing the corresponding predicted values for each example
        '''
        # Multiplying the weights by the X values to create a 1-d array, with the predicted values
        # in each row
        return np.matmul(X, self.weights)

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]


class TwoLayerNN:

    def __init__(self, hidden_size, activation=sigmoid, activation_derivative=sigmoid_derivative):
        '''
        @attrs:
            activation: the activation function applied after the first layer
            activation_derivative: the derivative of the activation function. Used for training.
            hidden_size: The hidden size of the network (an integer)
            output_neurons: The number of outputs of the network
        '''
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.hidden_size = hidden_size

        # In this assignment, we will only use output_neurons = 1.
        self.output_neurons = 1


    def _get_layer2_bias_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the output bias, b2.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/db2, a numpy array of dimension: output_neurons by 1
        '''
        # TODO

        ## Transforming relevant terms into a one-dimensional array for ease of multiplication
        inputs = np.ndarray.flatten(x)
        layer1_bias = np.ndarray.flatten(layer1_bias)
        layer2_bias = np.ndarray.flatten(layer2_bias)

        # Matrix of inputs to the hidden layer
        hidden_layer_inputs = np.dot(layer1_weights, inputs)

        # Adding the bias term
        hidden_layer_inputs = hidden_layer_inputs + layer1_bias
        hidden_layer_inputs = sigmoid(hidden_layer_inputs)

        # Computing the output of the hidden layer * weights
        final_layer_output = np.dot(layer2_weights, hidden_layer_inputs)

        # Adding the bias term
        final_layer_output = final_layer_output + layer2_bias

        gradient = (y - final_layer_output) * -2
        gradient = np.reshape(gradient, (1,1) )

        return gradient
        ## Finding the output

    def _get_layer2_weights_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the output weights, v.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/dv, a numpy array of dimension: output_neurons by hidden_size
        '''
        # TODO
        ## Transforming relevant terms into a one-dimensional array for ease of multiplication
        inputs = np.ndarray.flatten(x)
        layer1_bias = np.ndarray.flatten(layer1_bias)
        layer2_bias = np.ndarray.flatten(layer2_bias)

        # Matrix of inputs to the hidden layer
        hidden_layer_inputs = np.dot(layer1_weights, inputs)

        # Adding the bias term
        hidden_layer_inputs = hidden_layer_inputs + layer1_bias
        hidden_layer_inputs = sigmoid(hidden_layer_inputs)

        # Computing the output of the hidden layer * weights
        final_layer_output = np.dot(layer2_weights, hidden_layer_inputs)

        # Adding the bias term
        final_layer_output = final_layer_output + layer2_bias

        # Computing the actual gradient
        gradient = np.zeros(layer2_weights.size)

        gradient = hidden_layer_inputs * (-2) * (y - final_layer_output)

        # Reshaping into the desired size
        gradient = np.reshape (gradient, (1, layer2_weights.size))

        return gradient

    def _get_layer1_bias_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the hidden bias, b1.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/db1, a numpy array of dimension: hidden_size by 1
        '''
        # TODO

        ## Transforming relevant terms into a one-dimensional array for ease of multiplication
        inputs = np.ndarray.flatten(x)
        layer1_bias = np.ndarray.flatten(layer1_bias)
        layer2_bias = np.ndarray.flatten(layer2_bias)

        # Matrix of inputs to the hidden layer
        hidden_layer_inputs = np.dot(layer1_weights, inputs)

        # Adding the bias term
        hidden_layer_inputs = hidden_layer_inputs + layer1_bias
        hidden_layer_outputs = sigmoid(hidden_layer_inputs)

        # Computing the output of the hidden layer * weights
        final_layer_output = np.dot(layer2_weights, hidden_layer_outputs)

        # Adding the bias term
        final_layer_output = final_layer_output + layer2_bias



        # Computing the gradient
        gradient = np.multiply (layer2_weights, sigmoid_derivative(hidden_layer_inputs))
        gradient = gradient * (-2) * (y - final_layer_output)
        gradient = gradient.reshape((self.hidden_size, 1))

        return gradient


        pass

    def _get_layer1_weights_gradient(self, x, y, layer1_weights, layer1_bias,
        layer2_weights, layer2_bias):
        '''
        Computes the gradient of the loss with respect to the hidden weights, W.

        :param x: Numpy array for a single training example with dimension: input_size by 1
        :param y: Label for the training example
        :param layer1_weights: Numpy array of dimension: hidden_size by input_size
        :param layer1_bias: Numpy array of dimension: hidden_size by 1
        :param layer2_weights: Numpy array of dimension: output_neurons by hidden_size
        :param layer2_bias: Numpy array of dimension: output_neurons by 1
        :return: the partial derivates dL/dW, a numpy array of dimension: hidden_size by input_size
        '''
        ## Transforming relevant terms into a one-dimensional array for ease of multiplication
        inputs = np.ndarray.flatten(x)
        layer1_bias = np.ndarray.flatten(layer1_bias)
        layer2_bias = np.ndarray.flatten(layer2_bias)

        # Matrix of inputs to the hidden layer
        hidden_layer_inputs = np.dot(layer1_weights, inputs)

        # Adding the bias term
        hidden_layer_inputs = hidden_layer_inputs + layer1_bias
        hidden_layer_outputs = sigmoid(hidden_layer_inputs)

        # Computing the output of the hidden layer * weights
        final_layer_output = np.dot(layer2_weights, hidden_layer_outputs)

        # Adding the bias term
        final_layer_output = final_layer_output + layer2_bias


        ## Computing the gradient
        gradient = np.transpose (layer2_weights) * (-2) * (y - final_layer_output)
        hidden_derivative = sigmoid_derivative (hidden_layer_inputs)
        hidden_derivative = hidden_derivative.reshape((hidden_derivative.size, 1))
        gradient = np.multiply(gradient, hidden_derivative)
        gradient = np.multiply(gradient, np.transpose(x))
        return gradient

    def train(self, X, Y, learning_rate=0.01, epochs=1000, print_loss=True):
        '''
        Trains the TwoLayerNN with SGD using Backpropagation.

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :param learning_rate: The learning rate to use for SGD
        :param epochs: The number of times to pass through the dataset
        :param print_loss: If True, print the loss after each epoch.
        :return: None
        '''
        # NOTE:
        # Use numpy arrays of the following dimensions for your model's parameters.
        # layer 1 weights: hidden_size x input_size
        # layer 1 bias: hidden_size x 1
        # layer 2 weights: output_neurons x hidden_size
        # layer 2 bias: output_neurons x 1
        # HINT: for best performance initialize weights with np.random.normal or np.random.uniform
        # TODO
        input_size = X.shape[1]
        num_examples = Y.size

        self.layer1_weights = np.random.normal (size = (self.hidden_size, input_size))
        self.layer1_bias = np.random.normal (size = (self.hidden_size, 1))
        self.layer2_weights = np.random.normal (size = (1, self.hidden_size))
        self.layer2_bias = np.random.normal (size = (1, 1))
        for i in range (epochs):

            for j in range (num_examples):

                layer2_bias_gradient = self._get_layer2_bias_gradient(X[j], Y[j], self.layer1_weights, self.layer1_bias, self.layer2_weights, self.layer2_bias)
                layer2_weights_gradient = self._get_layer2_weights_gradient(X[j], Y[j], self.layer1_weights, self.layer1_bias, self.layer2_weights, self.layer2_bias)
                layer1_bias_gradient = self._get_layer1_bias_gradient(X[j], Y[j], self.layer1_weights, self.layer1_bias, self.layer2_weights, self.layer2_bias)
                layer1_weights_gradient = self._get_layer1_weights_gradient(X[j], Y[j], self.layer1_weights, self.layer1_bias, self.layer2_weights, self.layer2_bias)

                self.layer2_bias = self.layer2_bias - (learning_rate * layer2_bias_gradient )
                self.layer2_weights = self.layer2_weights- (learning_rate * layer2_weights_gradient )
                self.layer1_bias = self.layer1_bias - (learning_rate * layer1_bias_gradient )
                self.layer1_weights = self.layer1_weights- (learning_rate * layer1_weights_gradient )

            if (print_loss):
                print (self.average_loss(X, Y))

        pass

    def predict(self, X):
        '''
        Returns predictions of the model on a set of examples X.

        :param X: 2D Numpy array where each row contains an example.
        :return: A 1D Numpy array containing the corresponding predicted values for each example
        '''
        num_examples = X.shape[0]
        predictions = np.zeros(num_examples)
        layer1_bias = np.ndarray.flatten(self.layer1_bias)
        layer2_bias = np.ndarray.flatten(self.layer2_bias)

        for i in range (num_examples):
            ## Transforming relevant terms into a one-dimensional array for ease of multiplication
            inputs = X[i]
            # Matrix of inputs to the hidden layer
            hidden_layer_inputs = np.dot(self.layer1_weights, inputs)

            # Adding the bias term
            hidden_layer_inputs = hidden_layer_inputs + layer1_bias
            hidden_layer_inputs = sigmoid(hidden_layer_inputs)

            # Computing the output of the hidden layer * weights
            final_layer_output = np.dot(self.layer2_weights, hidden_layer_inputs)

            # Adding the bias term
            final_layer_output = final_layer_output + layer2_bias
            predictions[i] = final_layer_output

        return predictions

    def loss(self, X, Y):
        '''
        Returns the total squared error on some dataset (X, Y).

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the squared error of the model on the dataset
        '''
        predictions = self.predict(X)
        return l2_loss(predictions, Y)

    def average_loss(self, X, Y):
        '''
        Returns the mean squared error on some dataset (X, Y).

        MSE = Total squared error/# of examples

        :param X: 2D Numpy array where each row contains an example
        :param Y: 1D Numpy array containing the corresponding values for each example
        :return: A float which is the mean squared error of the model on the dataset
        '''
        return self.loss(X, Y)/X.shape[0]
