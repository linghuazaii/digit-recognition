#!/bin/env python
# -*- coding: utf-8 -*-
# This file is auto-generated.Edit it at your own peril.
import numpy as np
import random
from mnist_loader import *

def sigmoid(z):
    '''
    # there is an overflow for np.exp()
    val = 1.0 / (1.0 + np.exp(-z))
    # make it good for optimizors
    for (x, y), value in np.ndenumerate(val):
        if value >= 1.0 - 1.0e-10:
            val[x][y] = 1.
        elif value <= 1.0e-10:
            val[x][y] = 0.
    '''
    for (x, y), val in np.ndenumerate(z):
        if val >= 100:
            z[x][y] = 1.
        elif val <= -100:
            z[x][y] = 0.
        else:
            z[x][y] = 1.0 / (1.0 + np.exp(-val))

    return z

def sigmoid_derivative(z):
    return sigmoid(z) * (1. - sigmoid(z))

class NeuralNetwork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def describe(self):
        print 'layers:', self.num_layers
        print 'sizes:', self.sizes
        print 'biases:', self.biases
        print 'weights:', self.weights

    def feedForward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # eta is learning rate
    def stochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta, lm, test_data = None):
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in xrange(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, eta, lm, n_train)
            if test_data:
                print 'Epoch %s: %s / %s' % (j, self.evaluate(test_data), n_test)
            else:
                print 'Epoch %s complete.' % j

    def updateMiniBatch(self, mini_batch, eta, lm, total_train):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprob(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1.0 - eta * lm / total_train) * w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprob(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) # * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sd
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedForward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

def main():
    nn = NeuralNetwork([784, 100, 10])
    train = load_train_data()
    test = load_test_data()
    nn.stochasticGradientDescent(train, 60, 100, 1.0, 10.0, test_data = test)

if __name__ == "__main__":
    main()

