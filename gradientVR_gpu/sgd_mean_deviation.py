import numpy as np
import sys
import lasagne
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import main
import pickle

n = 10
num_epochs = 5

a = []
for i in range(n):
    temp = main.run_method(lasagne.updates.sgd, num_epochs=num_epochs)
    a.append(temp['grads'])

aa = []
mu = np.mean(a, axis=0)
for x in a:
    temp = []
    for i in range(n):
        angle = np.arccos(mu[i].dot(x[i]) / np.linalg.norm(x[i]) / np.linalg.norm(mu[i]))
        temp.append(angle)
    aa.append(temp)
aa = np.mean(aa, axis=0)


with open("sgd_mean_dev", "wb") as f:
    pickle.dump(aa, f)