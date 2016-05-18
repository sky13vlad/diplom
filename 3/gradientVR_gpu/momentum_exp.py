import numpy as np
import sys
import main
import lasagne
import pickle

momentum = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.momentum, num_epochs=50)
    momentum.append(temp['angle'])

with open('momentum_angle_batched', 'wb') as f:
    pickle.dump(momentum, f)
