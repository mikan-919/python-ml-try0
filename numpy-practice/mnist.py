import os.path as path
import pickle

def load_mnist():
  mnist_pickle_path=path.join(path.dirname(__file__),'mnist.pickle')
  if path.isfile(mnist_pickle_path):
    with open(mnist_pickle_path, 'rb') as f:
      mnist_dataset = pickle.load(f)
  else:
    from keras.datasets import mnist
    mnist_dataset = mnist.load_data()
    with open(mnist_pickle_path, 'wb') as f:
        pickle.dump(mnist_dataset, f)
  return mnist_dataset