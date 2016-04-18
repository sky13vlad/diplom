import numpy as np
import sys
import main
import lasagne
import pickle

momentum = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.momentum, num_epochs=50, BN=True)
    momentum.append(temp['angle'])

with open('bn_momentum_angle', 'wb') as f:
    pickle.dump(momentum, f)
