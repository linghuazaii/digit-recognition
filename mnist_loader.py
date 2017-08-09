#!/bin/env python
# -*- coding: utf-8 -*-
# This file is auto-generated.Edit it at your own peril.
import struct, gzip
import numpy as np
import Image

def vectorize_result(labels):
    result = list()
    for i in labels:
        rs = np.zeros((10, 1))
        rs[i][0] = 1.
        result.append(rs)
    return result

def load_train_data():
    fimg = gzip.open('train-images-idx3-ubyte.gz', 'rb')
    flabel = gzip.open('train-labels-idx1-ubyte.gz', 'rb')
    magic_img, total_img, rows, cols = struct.unpack('>IIII', fimg.read(16))
    magic_label, total_label = struct.unpack('>II', flabel.read(8))
    # print magic_label, total_label
    # print magic, total_img, rows, cols
    '''
    for i in xrange(1, 11):
        img = np.fromstring(fimg.read(rows * cols), dtype = np.uint8)
        img = Image.fromarray(np.reshape(img, (rows, cols)))
        label = struct.unpack('>B', flabel.read(1))[0]
        img_name = 'digits/digit%s_%s.png' % (i, label)
        img.save(img_name)
    '''
    train_data = list()
    for i in xrange(total_img):
        img = np.reshape(np.fromstring(fimg.read(rows * cols), dtype = np.uint8), (rows * cols, 1))
        train_data.append(img)
    train_label = vectorize_result(np.fromstring(flabel.read(total_label), dtype = np.uint8))
    train = zip(train_data, train_label)

    fimg.close()
    flabel.close()

    return train

def load_test_data():
    fimg = gzip.open('t10k-images-idx3-ubyte.gz', 'rb')
    flabel = gzip.open('t10k-labels-idx1-ubyte.gz', 'rb')
    magic_img, total_img, rows, cols = struct.unpack('>IIII', fimg.read(16))
    magic_label, total_label = struct.unpack('>II', flabel.read(8))
    # print magic_label, total_label
    # print magic_img, total_img, rows, cols
    '''
    for i in xrange(1, 11):
        img = np.fromstring(fimg.read(rows * cols), dtype = np.uint8)
        img = Image.fromarray(np.reshape(img, (rows, cols)))
        label = struct.unpack('>B', flabel.read(1))[0]
        img_name = 'digits/digit%s_%s.png' % (i, label)
        img.save(img_name)
    '''
    test_data = list()
    for i in xrange(total_img):
        img = np.reshape(np.fromstring(fimg.read(rows * cols), dtype = np.uint8), (rows * cols, 1))
        test_data.append(img)
    test_label = np.fromstring(flabel.read(total_label), dtype = np.uint8)
    test = zip(test_data, test_label)

    fimg.close()
    flabel.close()

    return test

'''
def main():
    load_train_data()
    load_test_data()

if __name__ == "__main__":
    main()
'''
