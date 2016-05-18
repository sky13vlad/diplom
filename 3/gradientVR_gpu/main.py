import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


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


def build_mlp(HL, sz, input_var=None, BN=False):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, sz, sz),
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


def build_cnn(CL, HL, sz, input_var=None, BN=False):
    network = lasagne.layers.InputLayer(shape=(None, 1, sz, sz),
                                        input_var=input_var)
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


def mvec(w):
    return np.concatenate(map(np.ravel, w))


def run_method(method, dataset='MNIST', model='mlp', CL=2, HL=3, BN=False, num_epochs=50, alpha=0.1, mu=0.9,
               beta1=0.9, beta2=0.999, epsilon=1e-8, echo=False, batch_size=500):
    sz = 1
    if dataset == 'MNIST':
        X_train, y_train, X_val, y_val, X_test, y_test = load_dataset_mnist()
        sz = 28

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if echo:
        print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(HL, sz, input_var, BN)
    elif model == 'cnn':
        network = build_cnn(CL, HL, sz, input_var, BN)
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

    arr_angle = []
    arr_grad = []

    arr_cos = []

    for epoch in range(num_epochs):
        temp_arr_cos = []
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        mind = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            if mind % 25 == 0:
                temp_err = 0

                cur_params = lasagne.layers.get_all_param_values(network)
                # full_params = np.zeros(mvec(cur_params).shape)
                # for batch_full in iterate_minibatches(X_train, y_train, batch_size):
                #     lasagne.layers.set_all_param_values(network, cur_params)
                #     inputs, targets = batch_full
                #     temp_err += train_fn(inputs, targets)
                #     full_params_part = lasagne.layers.get_all_param_values(network)
                #     full_params += mvec(full_params_part) - mvec(cur_params)
                # full_params /= len(X_train) / batch_size
                temp_err = train_fn(X_train, y_train)
                full_params = lasagne.layers.get_all_param_values(network)
                lasagne.layers.set_all_param_values(network, cur_params)

            inputs, targets = batch
            err = train_fn(inputs, targets)
            acc = train_fn_acc(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            iter_arr_train_err.append(train_err / train_batches)
            iter_arr_train_acc.append(train_acc / train_batches * 100)

            if mind % 25 == 0:
                new_params = lasagne.layers.get_all_param_values(network)
                # x = full_params

                s = map(str, lasagne.layers.get_all_params(network))
                s = np.array(s)
                ind = np.where(s == 'W')[0]

                cur_params = [cur_params[i] for i in ind]
                full_params = [full_params[i] for i in ind]
                new_params = [new_params[i] for i in ind]

                x = mvec(full_params) - mvec(cur_params)
                y = mvec(new_params) - mvec(cur_params)
                td = x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y))
                if abs(np.linalg.norm(x)) < 1e-8 or abs(np.linalg.norm(y)) < 1e-8:
                    td = 1.0
                temp_arr_cos.append(td)
                # print("cos: " + str(td))
                angle = np.arccos(td)
                arr_angle.append(angle)
                arr_grad.append(y)
            mind += 1
        arr_cos.append(temp_arr_cos)

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

    res['angle'] = arr_angle
    res['grads'] = arr_grad

    res['cos'] = arr_cos

    return res