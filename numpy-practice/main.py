import numpy as np
from keras.datasets import mnist
from pprint import pprint

# データ取得
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
np.set_printoptions(linewidth=150)
# train_imagesの配列を見る
print("配列の軸数：" + str(train_images.ndim))
print(train_images[0])
