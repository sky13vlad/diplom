import numpy as np
import sys
import main
import lasagne
import pickle

adam = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.adam, num_epochs=50)
    adam.append(temp['angle'])

with open('adam_angle', 'wb') as f:
    pickle.dump(adam, f)
