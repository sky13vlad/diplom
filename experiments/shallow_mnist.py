import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


def load_dataset():
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


def build_mlp(input_var=None, BN=False):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)
    if BN:
        l_in = lasagne.layers.batch_norm(l_in)

    l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    if BN:
        l_hid1 = lasagne.layers.batch_norm(l_hid1)

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)
    if BN:
        l_hid2 = lasagne.layers.batch_norm(l_hid2)

    l_out = lasagne.layers.DenseLayer(
            l_hid2, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    return l_out


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


def run_method(method, model='mlp', BN=False, num_epochs=50, alpha=0.1, mu=0.9, beta1=0.9, echo=False, batch_size=500):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if echo:
        print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var, BN)
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
        updates = method(loss, params, learning_rate=alpha, beta1=beta1)
    else:
        updates = method(loss, params, learning_rate=alpha)

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

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            err = train_fn(inputs, targets)
            acc = train_fn_acc(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1

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

        arr_val_err.append(val_err / val_batches)
        arr_val_acc.append(val_acc / val_batches * 100)

        if echo:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
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

    return res