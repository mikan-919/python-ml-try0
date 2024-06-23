from lib import mnist, common, two_layer_net
import numpy as np

np.set_printoptions(linewidth=300)

(train_image, train_label), (test_image, test_label) = mnist.load_mnist(
    flatten=True, normalize=True, one_hot=True
)
train_loss_list = []

iters_num = 10000
train_size = train_image.shape[0]
batch_size = 100
learning_rate = 0.1

network = two_layer_net.TwoLayerNet(784, 50, 10)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    batch_image = train_image[batch_mask]
    batch_label = train_label[batch_mask]
    print("iter: " + i + " ---------------------------------")
    grad = network.numerical_gradient(batch_image, batch_label)

    for key in ("w1", "b1", "w2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(batch_image, batch_label)
    print("Loss: " + loss)
    train_loss_list.append(loss)
