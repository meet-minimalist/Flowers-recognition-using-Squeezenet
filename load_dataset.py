import numpy as np
import tensorflow as tf
import glob
import sys
import cv2
import matplotlib.image as mpimg

class loadDataset():
    def __init__(self, train_split, test_split, valid_split, dataset_loc, train_op_fn, test_op_fn, valid_op_fn, isFixedSize=False, resize_dims=(224, 224)):
        self.all_files = glob.glob(dataset_loc)
        np.random.shuffle(self.all_files)            # shuffle the file names
        self.train_files = self.all_files[:int(len(self.all_files) * train_split)]
        self.test_files = self.all_files[int(len(self.all_files) * train_split) : int(len(self.all_files) * (train_split + test_split))]
        self.valid_files = self.all_files[int(len(self.all_files) * (train_split + test_split)):]
        self.isFixedSize = isFixedSize
        self.resize_size = resize_dims
        self.train_records_filename = train_op_fn
        self.test_records_filename = test_op_fn
        self.valid_records_filename = valid_op_fn
        self.no_channels = 3
        
    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def get_mean_rgb(self):
        r_mean = []
        g_mean = []
        b_mean = []
        for i in range(len(self.all_files)):
            img = mpimg.imread(self.all_files[i])
            r_mean.append(np.mean(img[:, :, 0]))
            g_mean.append(np.mean(img[:, :, 1]))
            b_mean.append(np.mean(img[:, :, 2]))
        self.r_mean = np.mean(r_mean)
        self.g_mean = np.mean(g_mean)
        self.b_mean = np.mean(b_mean)
    
    def get_label(self, addr):
        if 'daisy' in addr:
            label = 0
            return label
        elif 'dandelion' in addr:
            label = 1
            return label
        elif 'rose' in addr:
            label = 2
            return label
        elif 'sunflower' in addr:
            label = 3
            return label
        elif 'tulip' in addr:
            label = 4
            return label
    
    def load_image(self, addr):
        img = cv2.imread(addr)
        if img is None:
            return None
        img = cv2.resize(img, (self.resize_size[1], self.resize_size[0]), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img.shape[0], img.shape[1]
    
    def createDataRecord(self, out_filename, files):
        writer = tf.python_io.TFRecordWriter(out_filename)

        for i in range(len(files)):
            if not i % 1000:
                print('Processing completed : {} / {}'.format(i, len(files)))
                sys.stdout.flush()

            img, h, w = self.load_image(files[i])
            label = self.get_label(files[i])

            if self.isFixedSize:
                with tf.gfile.FastGFile(files[i], 'rb') as fid:
                    image_data = fid.read()
                feature = {'image' : self._bytes_feature(image_data),
                       'filename' : self._bytes_feature(files[i].encode('utf-8')),
                       'label' : self._int64_feature(label), 
                       'height' : self._int64_feature(h),
                       'width' : self._int64_feature(w)}
            else:
                feature = {'image' : self._bytes_feature(img.tostring()),
                       'filename' : self._bytes_feature(files[i].encode('utf-8')),
                       'label' : self._int64_feature(label), 
                       'height' : self._int64_feature(h),
                       'width' : self._int64_feature(w)}

            example = tf.train.Example(features = tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
        writer.close()
        sys.stdout.flush()
                
    def createDataRecordAll(self):
        self.createDataRecord(self.train_records_filename, self.train_files)
        self.createDataRecord(self.test_records_filename, self.test_files)
        self.createDataRecord(self.valid_records_filename, self.valid_files)
    
    def _extractFxn(self, tfrecord_file):
        features = {'image' : tf.FixedLenFeature([], tf.string),
                    'filename' : tf.FixedLenFeature([], tf.string),
                    'label' : tf.FixedLenFeature([], tf.int64),
                    'height' : tf.FixedLenFeature([], tf.int64),
                    'width' : tf.FixedLenFeature([], tf.int64)}
        sample = tf.parse_single_example(tfrecord_file, features)
        if self.isFixedSize:
            image = tf.image.decode_image(sample['image'], dtype=tf.float32) / 255
        else:
            image = tf.decode_raw(sample['image'], tf.uint8)
            image = tf.cast(image, tf.float32)
            image = tf.reshape(image, shape=[self.resize_size[0], self.resize_size[1], self.no_channels])
            img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
            image_rgb = tf.concat(axis=2, values=[
                                    img_r - self.r_mean,
                                    img_g - self.g_mean,
                                    img_b - self.b_mean])
            image_rgb = image_rgb / 255
        label = sample['label']
        one_hot_label = tf.one_hot(label, 5)        
        fn = sample['filename']
        return image, one_hot_label, fn
        
    def getTrainBatches(self, tfrecord_file, batch_size, isTrain=True):
        self.get_mean_rgb()
        dataset = tf.data.TFRecordDataset([tfrecord_file])
        dataset = dataset.map(self._extractFxn)
        
        if isTrain:
            dataset = dataset.shuffle(buffer_size=2048)
            num_repeat = None
        else:
            num_repeat = 1
            
        dataset = dataset.repeat(num_repeat)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_image_batch, next_label_batch, next_filenames_batch = iterator.get_next()
        
        return next_image_batch, next_label_batch, next_filenames_batch