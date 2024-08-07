import numpy as np
from lib import mnist, twolayernet
import matplotlib.pyplot as plt
from rich.progress import track
from math import ceil

np.set_printoptions(linewidth=300, precision=5)

(train_image, train_label), (test_image, test_label) = mnist.load_mnist(
    flatten=True, normalize=True, one_hot=True
)

iteration_num = 10000
batch_size = 300
learning_rate = 0.5
train_size = train_image.shape[0]
epoch_size = int(max(train_size / batch_size, 1))
epoch_iter = ceil(iteration_num / epoch_size)
learning_rate_decrase_rate = learning_rate / epoch_iter
print("Epoch iter: " + str(epoch_iter))
print("Epoch size: " + str(epoch_size))
print("Learning Rate: " + str(learning_rate))

train_loss_list = []
train_acc_list = []
test_acc_list = []
network_shape = (784, 28, 10)
network = twolayernet.TwoLayerNet(*network_shape)
print("Network Shape: " + str(network_shape))
print("======== ======== ======== ======== ======== ======== ======== ")

for i in track(range(epoch_iter), description="Training..."):
    for j in range(epoch_size):
        batch_mask = np.random.choice(train_size, batch_size)
        batch_image = train_image[batch_mask]
        batch_label = train_label[batch_mask]

        grad = network.gradient(batch_image, batch_label)

        for key in ("W1", "B1", "W2", "B2"):
            network.params[key] -= learning_rate * grad[key]

        loss = network.loss(batch_image, batch_label)
        train_loss_list.append(loss)

    # Epoch Acc Check
    train_acc = network.accuracy(train_image, train_label)
    test_acc = network.accuracy(test_image, test_label)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(
        "Accuracy(test:train) -> "
        + str(format(test_acc, ".3f"))
        + ":"
        + str(format(train_acc, ".3f"))
    )
    learning_rate -= learning_rate_decrase_rate


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
