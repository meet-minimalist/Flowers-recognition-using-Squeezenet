import tensorflow as tf
import numpy as np

no_classes = 1000

class SqueezeNet():
    def __init__ (self, no_classes):
        self.classes = no_classes
    
    def model(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        self.conv1 = tf.layers.conv2d(self.x, filters=96, kernel_size=(7,7), padding='same', strides=2, activation=tf.nn.relu)
        self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=3, strides=2)
        self.fire2 = self.fire_module(self.maxpool1, 16, 64, 64)
        self.fire3 = self.fire_module(self.fire2, 16, 64, 64)
        self.fire4 = self.fire_module(self.fire3, 32, 128, 128)
        self.maxpool4 = tf.layers.max_pooling2d(self.fire4, pool_size=3, strides=2)
        self.fire5 = self.fire_module(self.maxpool4, 32, 128, 128)
        self.fire6 = self.fire_module(self.fire5, 48, 192, 192)
        self.fire7 = self.fire_module(self.fire6, 48, 192, 192)
        self.fire8 = self.fire_module(self.fire7, 64, 256, 256)
        self.maxpool8 = tf.layers.max_pooling2d(self.fire8, pool_size=3, strides=2)
        self.fire9 = self.fire_module(self.maxpool8, 64, 256, 256)
        self.conv10 = tf.layers.conv2d(self.fire9, filters=self.classes, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)
        
        self.globalavgpool10 = tf.reduce_mean(self.conv10, axis=[1, 2])
        self.logits_gap = self.globalavgpool10
        
        return self.logits_gap
        
    def fire_module(input_, s11_filters, e11_filters, e33_filters):
        s1x1 = tf.layers.conv2d(input_, filters=s11_filters, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)    
        e1x1 = tf.layers.conv2d(s1x1, filters=e11_filters, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)
        e3x3 = tf.layers.conv2d(s1x1, filters=e33_filters, kernel_size=(3,3), padding='same', strides=1, activation=tf.nn.relu)
        fire = tf.concat([e1x1, e3x3], axis=3)
        return fire

