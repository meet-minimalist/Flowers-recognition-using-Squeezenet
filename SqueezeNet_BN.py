import tensorflow as tf
import numpy as np

class SqueezeNet():
    def __init__ (self, no_classes=1000, resize_dims=(224, 224)):
        self.classes = no_classes
        self.resize_shape = resize_dims
        tf.reset_default_graph()
        self.graph_context = tf.get_default_graph()
        self.leaky_relu_alpha = 0.02
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.resize_shape[0], self.resize_shape[1], 3])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.classes])
    
    def loadModel(self, reuse_existing_model=False, isTraining=False):
        with self.graph_context.as_default():            
            with tf.variable_scope("Conv1", reuse_existing_model):
                self.conv1 = tf.layers.conv2d(self.x, filters=96, kernel_size=(7,7), padding='same', strides=2, activation=None, use_bias=False)
                self.conv1 = tf.layers.batch_normalization(self.conv1, training=isTraining)
                self.conv1 = tf.maximum(self.leaky_relu_alpha  * self.conv1, self.conv1)
                self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=3, strides=2)
            with tf.variable_scope("Fire2", reuse_existing_model):
                self.fire2 = self.fire_module(self.maxpool1, 16, 64, 64, isTraining)
            with tf.variable_scope("Fire3", reuse_existing_model):
                self.fire3 = self.fire_module(self.fire2, 16, 64, 64, isTraining)
            with tf.variable_scope("Fire4", reuse_existing_model):
                self.fire4 = self.fire_module(self.fire3, 32, 128, 128, isTraining)
                self.fire4_drop = tf.layers.dropout(self.fire4)
                self.maxpool4 = tf.layers.max_pooling2d(self.fire4_drop, pool_size=3, strides=2)
            with tf.variable_scope("Fire5", reuse_existing_model):                
                self.fire5 = self.fire_module(self.maxpool4, 32, 128, 128, isTraining)
            with tf.variable_scope("Fire6", reuse_existing_model):
                self.fire6 = self.fire_module(self.fire5, 48, 192, 192, isTraining)
            with tf.variable_scope("Fire7", reuse_existing_model):
                self.fire7 = self.fire_module(self.fire6, 48, 192, 192, isTraining)
            with tf.variable_scope("Fire8", reuse_existing_model):                
                self.fire8 = self.fire_module(self.fire7, 64, 256, 256, isTraining)
                self.fire8_drop = tf.layers.dropout(self.fire8)
                self.maxpool8 = tf.layers.max_pooling2d(self.fire8_drop, pool_size=3, strides=2)
            with tf.variable_scope("Fire9", reuse_existing_model):                
                self.fire9 = self.fire_module(self.maxpool8, 64, 256, 256, isTraining)
                self.fire9_drop = tf.layers.dropout(self.fire9)
            with tf.variable_scope("Fire10", reuse_existing_model):                
                self.conv10 = tf.layers.conv2d(self.fire9_drop, filters=128, kernel_size=(1,1), padding='same', strides=1, activation=None, use_bias=False)
                self.conv10 = tf.layers.batch_normalization(self.conv10, training=isTraining)
                self.conv10 = tf.maximum(self.leaky_relu_alpha  * self.conv10, self.conv10)
                self.globalavgpool10 = tf.reduce_mean(self.conv10, axis=[1, 2])
            with tf.variable_scope("Output", reuse_existing_model):    
                self.logits_ap = tf.layers.dense(self.globalavgpool10, self.classes)        
                self.output = tf.nn.softmax(self.logits_ap)
        
    def fire_module(self, input_, s11_filters, e11_filters, e33_filters, isTraining):
        s1x1 = tf.layers.conv2d(input_, filters=s11_filters, kernel_size=(1,1), padding='same', strides=1, activation=None, use_bias=False)    
        s1x1 = tf.layers.batch_normalization(s1x1, training=isTraining)
        s1x1 = tf.maximum(self.leaky_relu_alpha  * s1x1, s1x1)
        e1x1 = tf.layers.conv2d(s1x1, filters=e11_filters, kernel_size=(1,1), padding='same', strides=1, activation=None, use_bias=False)
        e1x1 = tf.layers.batch_normalization(e1x1, training=isTraining)
        e1x1 = tf.maximum(self.leaky_relu_alpha  * e1x1, e1x1)
        e3x3 = tf.layers.conv2d(s1x1, filters=e33_filters, kernel_size=(3,3), padding='same', strides=1, activation=None, use_bias=False)
        e3x3 = tf.layers.batch_normalization(e3x3, training=isTraining)
        e3x3 = tf.maximum(self.leaky_relu_alpha  * e3x3, e3x3)
        fire = tf.concat([e1x1, e3x3], axis=3)
        return fire

    def train(self, epochs, b_size, tr_total, tr_x, tr_y, val_total, va_x, va_y, test_total, te_x, te_y, summary_path, model_save_path, lr=0.001):
        with self.graph_context.as_default():
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits_ap))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(self.y), tf.argmax(self.output)), tf.float32))
            self.loss_sum = tf.summary.scalar("Loss", self.loss)
            self.acc_sum = tf.summary.scalar("Accuray", self.accuracy)
            self.input_img = tf.summary.image("Input_image", self.x)
            self.merged_sum = tf.summary.merge_all()
        
        saver = tf.train.Saver()
        with tf.Session(graph=self.graph_context) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(summary_path)
            writer.add_graph(sess.graph)
            counter = 0
            for i in range(epochs):
                for j in range(tr_total // b_size):
                    train_x, train_y = sess.run([tr_x, tr_y])
                    _, l, merged = sess.run([self.opt, self.loss, self.merged_sum], feed_dict={self.x: train_x, self.y: train_y})
                    print("Epoch : {}/{}. Batch No. : {}/{}. Loss: {}".format(i+1, epochs, j+1, tr_total // b_size, l))
                    writer.add_summary(merged, counter)
                    counter += 1
                for k in range(val_total // b_size):
                    val_x, val_y = sess.run([va_x, va_y])
                    val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict={self.x:val_x, self.y:val_y})
                print("Epoch : {}/{} completed. Validation Loss : {}, Accuracy: {}".format(i+1, epochs, val_loss, val_acc * 100 / b_size))
                saver.save(sess, model_save_path)
            print("Training completed.")
            test_acc = []
            for m in range(test_total // b_size):
                test_x, test_y = sess.run([te_x, te_y])
                test_acc = sess.run(self.accuracy, feed_dict={self.x:test_x, self.y:test_y})
            print("Test dataset accuracy: {}".format(test_acc * 100 / b_size))
