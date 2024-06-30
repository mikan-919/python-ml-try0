from math import ceil
from lib_old import mnist, two_layer_net
import numpy as np
from rich.progress import track
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=300, precision=3)

(train_image, train_label), (test_image, test_label) = mnist.load_mnist(
    flatten=True, normalize=True, one_hot=True
)

iteration_num = 10000
batch_size = 100
learning_rate = 0.1
train_size = train_image.shape[0]
epoch_size = int(max(train_size / batch_size, 1))
epoch_iter = ceil(iteration_num / epoch_size)
print("Epoch iter: " + str(epoch_iter))
print("Batch size: " + str(batch_size))
print("Train size: " + str(train_size))
print("Epoch size: " + str(epoch_size))
print("======== ======== ======== ======== ======== ======== ======== ")

train_loss_list = []
train_acc_list = []
test_acc_list = []
network = two_layer_net.TwoLayerNet(784, 20, 10)

for i in range(epoch_iter):
    for j in track(range(epoch_size), description="Epoch-" + str(i)):
        batch_mask = np.random.choice(train_size, batch_size)
        batch_image = train_image[batch_mask]
        batch_label = train_label[batch_mask]

        grad = network.gradient(batch_image, batch_label)

        for key in ("w1", "b1", "w2", "b2"):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(batch_image, batch_label)
        train_loss_list.append(loss)

    # Epoch Acc Check
    train_acc = network.accuracy(train_image, train_label)
    test_acc = network.accuracy(test_image, test_label)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("Accuracy: " + str(test_acc))


x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label="loss")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.show()

x2 = np.arange(len(train_acc_list))
plt.plot(x2, train_acc_list, label="train acc")
plt.plot(x2, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.xlim(left=0)
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()
