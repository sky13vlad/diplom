import numpy as np
import sys
import main
import lasagne
import pickle

sgd = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.sgd, num_epochs=50)
    sgd.append(temp['angle'])

with open('sgd_angle_batched', 'wb') as f:
    pickle.dump(sgd, f)
