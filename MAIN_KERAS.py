import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
from keras.preprocessing.image import ImageDataGenerator

def load_dataset_mnist():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset_cifar10():
    import pickle

    with open('cifar-10-batches-py/data_batch_1', 'rb') as f:
        dict1 = pickle.load(f)
    with open('cifar-10-batches-py/data_batch_2', 'rb') as f:
        dict2 = pickle.load(f)
    with open('cifar-10-batches-py/data_batch_3', 'rb') as f:
        dict3 = pickle.load(f)
    with open('cifar-10-batches-py/data_batch_4', 'rb') as f:
        dict4 = pickle.load(f)
    with open('cifar-10-batches-py/data_batch_5', 'rb') as f:
        dict5 = pickle.load(f)

    with open('cifar-10-batches-py/test_batch', 'rb') as f:
        dict_test = pickle.load(f)

    data1 = dict1['data']
    data2 = dict2['data']
    data3 = dict3['data']
    data4 = dict4['data']
    data5 = dict5['data']

    data_train = np.concatenate((data1, data2, data3, data4, data5), axis=0)
    data_train = data_train.reshape(-1, 3, 32, 32) / np.float32(256)

    data_test = dict_test['data'].reshape(-1, 3, 32, 32) / np.float32(256)

    labels1 = dict1['labels']
    labels2 = dict2['labels']
    labels3 = dict3['labels']
    labels4 = dict4['labels']
    labels5 = dict5['labels']

    labels_train = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)
    labels_train = np.array(labels_train.ravel(), dtype=np.uint8)

    labels_test = np.array(np.ravel(dict_test['labels']), dtype=np.uint8)

    np.random.seed(42)
    inds = np.arange(len(labels_train))
    np.random.shuffle(inds)

    X_train, X_val = data_train[inds][:-10000], data_train[inds][-10000:]
    y_train, y_val = labels_train[inds][:-10000], labels_train[inds][-10000:]
    X_test = data_test
    y_test = labels_test

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_mlp(HL, sz, input_var=None, BN=False, channels=1):
    l_in = lasagne.layers.InputLayer(shape=(None, channels, sz, sz),
                                     input_var=input_var)
    if BN:
        l_in = lasagne.layers.batch_norm(l_in)

    l_prev = l_in

    for i in range(HL):
        l_hid = lasagne.layers.DenseLayer(
                l_prev, num_units=100,
                nonlinearity=lasagne.nonlinearities.rectify,
                W=lasagne.init.GlorotUniform())
        if BN:
            l_hid = lasagne.layers.batch_norm(l_hid)
        l_prev = l_hid

    l_out = lasagne.layers.DenseLayer(
            l_prev, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


def build_cnn(CL, HL, sz, input_var=None, BN=False, channels=1):
    network = lasagne.layers.InputLayer(shape=(None, channels, sz, sz),
                                        input_var=input_var)
    if BN:
        network = lasagne.layers.batch_norm(network)
    prev = network

    for i in range(CL):
        network = lasagne.layers.Conv2DLayer(
            prev, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        if BN:
            network = lasagne.layers.batch_norm(network)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        prev = network

    for i in range(HL):
        network = lasagne.layers.DenseLayer(
            prev, num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
        if BN:
            network = lasagne.layers.batch_norm(network)
        prev = network

    network = lasagne.layers.DenseLayer(
        prev, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def run_method(method, dataset='MNIST', model='mlp', CL=2, HL=3, BN=False, num_epochs=50, alpha=0.1, mu=0.9,
               beta1=0.9, beta2=0.999, epsilon=1e-8, echo=False, batch_size=500):
    sz = 1
    channels = 1
    if dataset == 'MNIST':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist()
        sz = 28
        channels = 1
    elif dataset == 'CIFAR-10':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_cifar10()
        sz = 32
        channels = 3

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if echo:
        print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(HL, sz, input_var, BN, channels)
    elif model == 'cnn':
        network = build_cnn(CL, HL, sz, input_var, BN, channels)
    else:
        print("Unrecognized model type %r." % model)
        return

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                 dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)

    if method == lasagne.updates.sgd:
        updates = method(loss, params, learning_rate=alpha)
    elif method == lasagne.updates.momentum:
        updates = method(loss, params, learning_rate=alpha, momentum=mu)
    elif method == lasagne.updates.adam:
        updates = method(loss, params, learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
    else:
        updates = method(loss, params, learning_rate=alpha, epsilon=epsilon)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_fn_acc = theano.function([input_var, target_var], train_acc)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if echo:
        print("Starting training...")

    res = dict()
    arr_train_err = []
    arr_val_err = []
    arr_train_acc = []
    arr_val_acc = []

    iter_arr_train_err = []
    iter_arr_val_err = []
    iter_arr_train_acc = []
    iter_arr_val_acc = []

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2)
    datagen.fit(X_train)

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()

        for batch in datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True):
            inputs, targets = batch
            inputs = np.array(inputs, dtype=np.float32)
            err = train_fn(inputs, targets)
            acc = train_fn_acc(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            iter_arr_train_err.append(train_err / train_batches)
            iter_arr_train_acc.append(train_acc / train_batches * 100)
            if train_batches >= len(X_train) / batch_size:
                break

        arr_train_err.append(train_err / train_batches)
        arr_train_acc.append(train_acc / train_batches * 100)

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
            iter_arr_val_err.append(val_err / val_batches)
            iter_arr_val_acc.append(val_acc / val_batches * 100)

        arr_val_err.append(val_err / val_batches)
        arr_val_acc.append(val_acc / val_batches * 100)

        if echo:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  train accuracy:\t\t{:.2f} %".format(train_acc / train_batches * 100))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    if echo:
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

    res['train_err'] = np.array(arr_train_err)
    res['val_err'] = np.array(arr_val_err)
    res['train_acc'] = np.array(arr_train_acc)
    res['val_acc'] = np.array(arr_val_acc)
    res['test_err'] = test_err / test_batches
    res['test_acc'] = test_acc / test_batches * 100

    res['iter_train_err'] = np.array(iter_arr_train_err)
    res['iter_val_err'] = np.array(iter_arr_val_err)
    res['iter_train_acc'] = np.array(iter_arr_train_acc)
    res['iter_val_acc'] = np.array(iter_arr_val_acc)

    return res
