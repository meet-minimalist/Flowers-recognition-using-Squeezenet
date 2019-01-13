from SqueezeNet_BN import SqueezeNet
from load_dataset import loadDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

batch_size = 8
resize_shape = (320, 240)
tr_split = 0.75
test_split = 0.1
val_split = 0.15
total_classes = 5
total_files = glob.glob("./flowers-recognition/flowers/*/*.jpg")
tr_size = int(len(total_files) * tr_split)
test_size = int(len(total_files) * test_split)
val_size = len(total_files) - tr_size - test_size
epochs = 10

# R_mean = 116.984, G_mean = 107.073, B_mean = 76.621

model = SqueezeNet(no_classes=5, resize_dims=resize_shape)
model.loadModel(reuse_existing_model=False, isTraining=True)
dataloader = loadDataset(tr_split, test_split, val_split, "./flowers-recognition/flowers/*/*.jpg", "train_resize.tfrecords", "test_resize.tfrecords", "valid_resize.tfrecords", resize_dims=resize_shape)

#dataloader.createDataRecordAll()

train_x, train_y, _ = dataloader.getTrainBatches("train_resize.tfrecords", batch_size, isTrain=True)
val_x, val_y, _ = dataloader.getTrainBatches("valid_resize.tfrecords", batch_size, isTrain=True)
test_x, test_y, _ = dataloader.getTrainBatches("test_resize.tfrecords", batch_size, isTrain=False)

model.train(20, batch_size, tr_size, train_x, train_y, val_size, val_x, val_y, test_size, test_x, test_y, "./summaries/4", "./saved_models/model1.ckpt", lr=0.0005)

model.train(20, batch_size, tr_size, train_x, train_y, val_size, val_x, val_y, test_size, test_x, test_y, "./summaries/5", "./saved_models/model2.ckpt", lr=0.00001)

model.train(20, batch_size, tr_size, train_x, train_y, val_size, val_x, val_y, test_size, test_x, test_y, "./summaries/6", "./saved_models/model3.ckpt", lr=0.00008)

model.train(20, batch_size, tr_size, train_x, train_y, val_size, val_x, val_y, test_size, test_x, test_y, "./summaries/7", "./saved_models/model4.ckpt", lr=0.00005)
    