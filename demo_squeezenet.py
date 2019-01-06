from SqueezeNet import SqueezeNet
from load_dataset import loadDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

batch_size = 16
tr_split = 0.75
test_split = 0.1
val_split = 0.15
total_classes = 5
total_files = glob.glob("./flowers-recognition/flowers/*/*.jpg")
tr_size = int(len(total_files) * tr_split)
test_size = int(len(total_files) * test_split)
val_size = len(total_files) - tr_size - test_size
epochs = 10

dataloader = loadDataset(tr_split, test_split, val_split, "./flowers-recognition/flowers/*/*.jpg", "train_resize.tfrecords", "test_resize.tfrecords", "valid_resize.tfrecords")

dataloader.createDataRecordAll()

train_x, train_y, _ = dataloader.getTrainBatches("train_resize.tfrecords", batch_size, isTrain=True)
test_x, test_y, _ = dataloader.getTrainBatches("test_resize.tfrecords", batch_size, isTrain=False)
val_x, val_y, _ = dataloader.getTrainBatches("valid_resize.tfrecords", batch_size, isTrain=True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    x, y = sess.run([train_x, train_y])


x.shape
y.shape




y_ = np.argmax(y, axis=1)

c = 0
for i in range(16):
    if y_[i] == 1:
        c += 1    
c

c_0 = 2, 3
c_1 = 4, 5
c_2 = 3, 1
c_3 = 4, 4
c_4 = 3, 3

model = SqueezeNet(no_classes=total_classes)
model.model()
model.train(10, batch_size, tr_size, train_x, train_y, val_size, val_x, val_y, test_size, test_x, test_y, lr=0.001)


