import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne

from collections import OrderedDict
from lasagne import utils


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


def adam_update(loss_or_grads, params, learning_rate=1e-3, beta1=0.9,
                        beta2=0.999, epsilon=1e-8):
    all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()
    u_v_min =  T.ones_like(1, dtype=theano.config.floatX)*100
    u_v_max =  T.zeros_like(1, dtype=theano.config.floatX)
    u_v_mean =  T.zeros_like(1, dtype=theano.config.floatX)
    tmp = T.zeros_like(1, dtype=theano.config.floatX)
    u_m_min =  T.ones_like(1, dtype=theano.config.floatX)*100
    u_m_max =  T.zeros_like(1, dtype=theano.config.floatX)
    u_m_mean =  T.zeros_like(1, dtype=theano.config.floatX)
    u_s_min =  T.ones_like(1, dtype=theano.config.floatX)*100
    u_s_max =  T.zeros_like(1, dtype=theano.config.floatX)
    u_s_mean =  T.zeros_like(1, dtype=theano.config.floatX)
    u_w_mean =  T.zeros_like(1, dtype=theano.config.floatX)

    bn_u_v_min = T.ones_like(1, dtype=theano.config.floatX) * 100
    bn_u_v_max = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_v_mean = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_m_min = T.ones_like(1, dtype=theano.config.floatX) * 100
    bn_u_m_max = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_m_mean = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_s_min = T.ones_like(1, dtype=theano.config.floatX) * 100
    bn_u_s_max = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_s_mean = T.zeros_like(1, dtype=theano.config.floatX)
    bn_u_w_mean = T.zeros_like(1, dtype=theano.config.floatX)
    bn_tmp = T.zeros_like(1, dtype=theano.config.floatX)

    t = t_prev + 1
    a_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon) #* T.minimum(1,(T.sqrt(v_t) + epsilon)/a_t)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

        s = str(param)
        if s == 'W':
            u_v_min =  T.minimum(T.min(v_t,axis=None), u_v_min)
            u_v_max =  T.maximum(T.max(v_t,axis=None), u_v_max)
            u_v_mean +=  T.sum(v_t,axis=None)
            tmp += T.prod(T.shape(v_t))

            u_m_min =  T.minimum(T.min(T.abs_(m_t),axis=None), u_m_min)
            u_m_max =  T.maximum(T.max(T.abs_(m_t),axis=None), u_m_max)
            u_m_mean +=  T.sum(T.abs_(m_t),axis=None)

            u_s_min =  T.minimum(T.min(T.abs_(g_t),axis=None), u_m_min)
            u_s_max =  T.maximum(T.max(T.abs_(g_t),axis=None), u_m_max)
            u_s_mean +=  T.sum(T.abs_(g_t),axis=None)
            u_w_mean += T.sum(T.abs_(param), axis=None)
        elif s == 'beta' or s == 'gamma' or s == 'mean' or s == 'inv_std':
            bn_u_v_min = T.minimum(T.min(v_t, axis=None), u_v_min)
            bn_u_v_max = T.maximum(T.max(v_t, axis=None), u_v_max)
            bn_u_v_mean += T.sum(v_t, axis=None)
            bn_tmp += T.prod(T.shape(v_t))

            bn_u_m_min = T.minimum(T.min(T.abs_(m_t), axis=None), u_m_min)
            bn_u_m_max = T.maximum(T.max(T.abs_(m_t), axis=None), u_m_max)
            bn_u_m_mean += T.sum(T.abs_(m_t), axis=None)

            bn_u_s_min = T.minimum(T.min(T.abs_(g_t), axis=None), u_m_min)
            bn_u_s_max = T.maximum(T.max(T.abs_(g_t), axis=None), u_m_max)
            bn_u_s_mean += T.sum(T.abs_(g_t), axis=None)
            bn_u_w_mean += T.sum(T.abs_(param), axis=None)



    updates[t_prev] = t
    return updates, u_v_min, u_v_max, u_v_mean/tmp,u_m_min, u_m_max, u_m_mean/tmp,u_s_min, u_s_max, u_s_mean/tmp,u_w_mean/tmp, \
           bn_u_v_min, bn_u_v_max, bn_u_v_mean / bn_tmp, bn_u_m_min, bn_u_m_max, bn_u_m_mean / bn_tmp, bn_u_s_min, bn_u_s_max, bn_u_s_mean / bn_tmp, bn_u_w_mean/bn_tmp


def run_method(method, model='mlp', BN=False, num_epochs=50, alpha=1e-3, mu=0.9, HL=3, beta1=0.9, beta2=0.999, epsilon = 1e-8,echo=False, batch_size=500):
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    if echo:
        print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(HL, 28, input_var, BN)
    else:
        print("Unrecognized model type %r." % model)
        return
    print("The network has {} params".format(lasagne.layers.count_params(network)))
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                 dtype=theano.config.floatX)

    params = lasagne.layers.get_all_params(network, trainable=True)
    '''
    if method == lasagne.updates.sgd:
        updates = method(loss, params, learning_rate=alpha)
    elif method == lasagne.updates.momentum:
        updates = method(loss, params, learning_rate=alpha, momentum=mu)
    elif method == lasagne.updates.adam:
        updates = method(loss, params, learning_rate=alpha, beta1=beta1)
    elif method == adam_update or method == adam_update2:
        updates = method(loss, params, learning_rate=alpha, beta1=beta1, beta2=beta2)
    else:
        updates = method(loss, params, learning_rate=alpha)
    '''
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    
    updates,u_v_min, u_v_max,u_v_mean,u_m_min, u_m_max, u_m_mean,u_s_min, u_s_max, u_s_mean,u_w_mean, bn_u_v_min, bn_u_v_max,bn_u_v_mean, \
            bn_u_m_min, bn_u_m_max, bn_u_m_mean, bn_u_s_min, bn_u_s_max, bn_u_s_mean, bn_u_w_mean = method(loss, params, learning_rate=alpha, beta1=beta1, beta2=beta2,epsilon = epsilon)
    train_fn = theano.function([input_var, target_var], [loss,u_v_min, u_v_max,u_v_mean,u_m_min, u_m_max,u_m_mean,u_s_min, u_s_max, u_s_mean,u_w_mean,
                                                         bn_u_v_min, bn_u_v_max, bn_u_v_mean, bn_u_m_min, bn_u_m_max, bn_u_m_mean, bn_u_s_min, bn_u_s_max,
                                                         bn_u_s_mean, bn_u_w_mean], updates=updates)
    train_fn_acc = theano.function([input_var, target_var], train_acc)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if echo:
        print("Starting training...")

    res = dict()
    arr_train_err = []
    arr_val_err = []
    arr_train_acc = []
    arr_val_acc = []
    v_min = []
    v_max = []
    v_mean = []
    m_min = []
    m_max = []
    m_mean = []
    s_min = []
    s_max = []
    s_mean = []
    w_mean = []

    bn_v_min = []
    bn_v_max = []
    bn_v_mean = []
    bn_m_min = []
    bn_m_max = []
    bn_m_mean = []
    bn_s_min = []
    bn_s_max = []
    bn_s_mean = []
    bn_w_mean = []

    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        train_acc = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            global inputs, targets
            inputs, targets = batch
            err,u_v_1, u_v_2,u_v_3,u_m_1, u_m_2, u_m_3,u_s_1, u_s_2, u_s_3,u_w_3, bn_u_v_1, bn_u_v_2,  bn_u_v_3, bn_u_m_1, bn_u_m_2, bn_u_m_3, bn_u_s_1, bn_u_s_2, bn_u_s_3, bn_u_w_3  = train_fn(inputs, targets)
            acc = train_fn_acc(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
            
            v_min.append(u_v_1)
            v_max.append(u_v_2)
            v_mean.append(u_v_3)
            m_min.append(u_m_1)
            m_max.append(u_m_2)
            m_mean.append(u_m_3)
            s_min.append(u_s_1)
            s_max.append(u_s_2)
            s_mean.append(u_s_3)
            w_mean.append(u_w_3)

            bn_v_min.append(bn_u_v_1)
            bn_v_max.append(bn_u_v_2)
            bn_v_mean.append(bn_u_v_3)
            bn_m_min.append(bn_u_m_1)
            bn_m_max.append(bn_u_m_2)
            bn_m_mean.append(bn_u_m_3)
            bn_s_min.append(bn_u_s_1)
            bn_s_max.append(bn_u_s_2)
            bn_s_mean.append(bn_u_s_3)
            bn_w_mean.append(bn_u_w_3)

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
    res['v_min'] = np.array(v_min)
    res['v_max'] = np.array(v_max)
    res['v_mean'] = np.array(v_mean)
    res['m_min'] = np.array(m_min)
    res['m_max'] = np.array(m_max)
    res['m_mean'] = np.array(m_mean)
    res['s_min'] = np.array(s_min)
    res['s_max'] = np.array(s_max)
    res['s_mean'] = np.array(s_mean)
    res['w_mean'] = np.array(w_mean)

    res['bn_v_min'] = np.array(bn_v_min)
    res['bn_v_max'] = np.array(bn_v_max)
    res['bn_v_mean'] = np.array(bn_v_mean)
    res['bn_m_min'] = np.array(bn_m_min)
    res['bn_m_max'] = np.array(bn_m_max)
    res['bn_m_mean'] = np.array(bn_m_mean)
    res['bn_s_min'] = np.array(bn_s_min)
    res['bn_s_max'] = np.array(bn_s_max)
    res['bn_s_mean'] = np.array(bn_s_mean)
    res['bn_w_mean'] = np.array(bn_w_mean)

    return res