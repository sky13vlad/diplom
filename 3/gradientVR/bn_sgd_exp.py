import main
import lasagne
import pickle

bn_sgd = []
for i in range(50):
    temp = main.run_method(lasagne.updates.sgd, num_epochs=50, BN=True)
    bn_sgd.append(temp['angle'])


with open('bn_sgd_angle', 'wb') as f:
    pickle.dump(bn_sgd, f)