#!/bin/env python
# -*- coding: utf-8 -*-
# This file is auto-generated.Edit it at your own peril.
import numpy as np
import random

def sigmoid(z):
    val = 1.0 / (1.0 + np.exp(-z))
    # make it good for optimizors
    for (x, y), value in np.ndenumerate(val):
        if val >= 1.0 - 1.0e-10:
            val[x][y] = 1.
        elif val <= 1.0e-10:
            val[x][y] = 0.
    return val

class NeuralNetwork(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

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
    def stochasticGradientDescent(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        if test_data:
            n_test = len(test_data)
        n_train = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print 'Epoch %s: %s / %s' % (j, self.evaluate(test_data), n_test)
            else:
                print 'Epoch %s complete.' % j

def main():
    nn = NeuralNetwork([5,8,4,2,1])
    nn.describe()

if __name__ == "__main__":
    main()

