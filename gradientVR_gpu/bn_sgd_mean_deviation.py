import numpy as np
import sys
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import main
import pickle

n = 25
num_epochs = 50

b = []
for i in range(n):
    temp = main.run_method(lasagne.updates.sgd, num_epochs=num_epochs, BN=True)
    b.append(temp['grads'])

bb = []
mu = np.mean(b, axis=0)
for x in b:
    temp = []
    for i in range(n):
        angle = np.arccos(mu[i].dot(x[i]) / np.linalg.norm(x[i]) / np.linalg.norm(mu[i]))
        temp.append(angle)
    bb.append(temp)
bb = np.mean(bb, axis=0)

with open("bn_sgd_mean_dev", "wb") as f:
    pickle.dump(bb, f)