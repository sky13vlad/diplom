import numpy as np
import sys
import main
import lasagne
import pickle

bn_momentum = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.momentum, num_epochs=50, BN=True)
    bn_momentum.append(temp['angle'])

with open('bn_momentum_angle_batched', 'wb') as f:
    pickle.dump(bn_momentum, f)
