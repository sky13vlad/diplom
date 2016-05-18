import main
import lasagne
import pickle

bn_adam = []
for i in range(50):
    print(i)
    temp = main.run_method(lasagne.updates.adam, num_epochs=50, BN=True)
    bn_adam.append(temp['angle'])


with open('bn_adam_angle', 'wb') as f:
    pickle.dump(bn_adam, f)
