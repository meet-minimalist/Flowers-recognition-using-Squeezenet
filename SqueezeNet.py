import tensorflow as tf
import numpy as np

no_classes = 1000

class SqueezeNet():
    def __init__ (self, no_classes):
        self.classes = no_classes
        self.graph_context = tf.get_default_graph()
    
    def model(self):
        with self.graph_context.as_default():            
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.classes])
    
            with tf.variable_scope("Conv1"):
                self.conv1 = tf.layers.conv2d(self.x, filters=96, kernel_size=(7,7), padding='same', strides=2, activation=tf.nn.relu)
                self.maxpool1 = tf.layers.max_pooling2d(self.conv1, pool_size=3, strides=2)
            with tf.variable_scope("Fire2"):
                self.fire2 = self.fire_module(self.maxpool1, 16, 64, 64)
            with tf.variable_scope("Fire3"):
                self.fire3 = self.fire_module(self.fire2, 16, 64, 64)
            with tf.variable_scope("Fire4"):
                self.fire4 = self.fire_module(self.fire3, 32, 128, 128)
                self.fire4_drop = tf.layers.dropout(self.fire4)
                self.maxpool4 = tf.layers.max_pooling2d(self.fire4_drop, pool_size=3, strides=2)
            with tf.variable_scope("Fire5"):                
                self.fire5 = self.fire_module(self.maxpool4, 32, 128, 128)
            with tf.variable_scope("Fire6"):
                self.fire6 = self.fire_module(self.fire5, 48, 192, 192)
            with tf.variable_scope("Fire7"):
                self.fire7 = self.fire_module(self.fire6, 48, 192, 192)
            with tf.variable_scope("Fire8"):                
                self.fire8 = self.fire_module(self.fire7, 64, 256, 256)
                self.fire8_drop = tf.layers.dropout(self.fire8)
                self.maxpool8 = tf.layers.max_pooling2d(self.fire8_drop, pool_size=3, strides=2)
            with tf.variable_scope("Fire9"):                
                self.fire9 = self.fire_module(self.maxpool8, 64, 256, 256)
                self.fire9_drop = tf.layers.dropout(self.fire9)
            with tf.variable_scope("Fire10"):                
                self.conv10 = tf.layers.conv2d(self.fire9_drop, filters=self.classes, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)
                self.avgpool10 = tf.layers.average_pooling2d(self.conv10, pool_size=13, strides=1)
            with tf.variable_scope("Output"):    
                self.logits_ap = tf.squeeze(self.avgpool10, axis=[1, 2])        
                self.output = tf.nn.softmax(self.logits_ap)
                
            #return self.logits_ap, self.output
 
    def fire_module(self, input_, s11_filters, e11_filters, e33_filters):
        with self.graph_context.as_default():            
            s1x1 = tf.layers.conv2d(input_, filters=s11_filters, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)    
            e1x1 = tf.layers.conv2d(s1x1, filters=e11_filters, kernel_size=(1,1), padding='same', strides=1, activation=tf.nn.relu)
            e3x3 = tf.layers.conv2d(s1x1, filters=e33_filters, kernel_size=(3,3), padding='same', strides=1, activation=tf.nn.relu)
            fire = tf.concat([e1x1, e3x3], axis=3)
            return fire

    def train(self, epochs, b_size, tr_total, tr_x, tr_y, val_total, va_x, va_y, test_total, te_x, te_y, lr=0.001):
        with self.graph_context.as_default():            
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.logits_ap))
            self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
        
        with tf.Session(graph=self.graph_context) as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("./summaries/3")
            writer.add_graph(sess.graph)
            
            for i in range(epochs):
                for j in range(tr_total // b_size):
                    train_x, train_y = sess.run([tr_x, tr_y])
                    print(np.argmax(train_y, axis=1))
                    _, l = sess.run([self.opt, self.loss], feed_dict={self.x: train_x, self.y: train_y})
                    print("Epoch : {}/{}. Batch No. : {}/{}. Loss: {}".format(i+1, epochs, j+1, tr_total // b_size, l))
                val_mean_loss = []
                val_mean_acc = []
                for k in range(val_total // b_size):
                    val_x, val_y = sess.run([va_x, va_y])
                    val_l, val_output = sess.run([self.loss, self.output], feed_dict={self.x:val_x, self.y:val_y})
                    val_mean_loss.append(val_l)
#                    print(np.argmax(val_y, axis=1), "********")
                    print(val_output, "************")
#                    print(np.argmax(val_output, axis=1), "~~~~~~~")
#                    print(np.equal(np.argmax(val_y, axis=1), np.argmax(val_output, axis=1)), "000000000")
                    acc = np.equal(np.argmax(val_y, axis=1), np.argmax(val_output, axis=1)).astype('uint8')
                    val_mean_acc.append(acc)
                print("Epoch : {}/{} completed. Mean Validation Loss : {}, Mean Accuracy: {}".format(i+1, epochs, np.mean(val_mean_loss), np.mean(val_mean_acc) * 100 / b_size))
            print("Training completed.")
            test_acc = []
            for m in range(test_total // b_size):
                test_x, test_y = sess.run([te_x, te_y])
                test_output = sess.run([self.output], feed_dict={self.x : test_x})
                acc = tf.cast(tf.equal(tf.argmax(test_y, axis=1), tf.argmax(test_output, axis=1)), dtype=tf.int8)
                test_acc.append(acc)
            print("Test dataset accuracy: {}".format(np.mean(test_acc) * 100 / b_size))
                        
                        
                        
                        
                
            