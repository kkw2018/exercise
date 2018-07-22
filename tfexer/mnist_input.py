import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import collections
import csv
import os
from os import path
import random
import time
import  gzip
from tensorflow.python.data import Dataset
import numpy as np
from six.moves import urllib
from sklearn import metrics
import glob

from tensorflow.python.platform import gfile
from tensorflow.python.util.deprecation import deprecated

'''
1.download data
2.input func
3.feature columns
4.DNN model
    optimizer,classfier,train,loss
5.train,predict,validate
'''

def maybe_download(filename, work_directory, source_url):
  """Download the data from source url, unless it's already here.

  Args:
      filename: string, name of the file in the directory.
      work_directory: string, path to working directory.
      source_url: url to download from if file doesn't exist.

  Returns:
      Path to resulting file.
  """
  if not gfile.Exists(work_directory):
      gfile.MakeDirs(work_directory)
  filepath=os.path.join(work_directory,filename)
  if not gfile.Exists(filepath):
      urllib.request.urlretrieve(source_url,filename=filepath)
  with gfile.GFile(filepath,'r') as f:
      size=f.size()
      print("Successfully downloaded",filename,size,"bytes.")
  return filepath



def mnist_download():
  """download minist files

  :return:
  """
  data_urls=[
      'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
      'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
  ]
  work_directory='e:/data'
  for source_url in data_urls:
    filename = source_url.split('/')[-1]
    maybe_download(filename, work_directory, source_url)


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def extract_images(f):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

  Args:
    f: A file object that can be passed into a gzip reader.

  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].

  Raises:
    ValueError: If the bytestream does not start with 2051.

  """
  print("Extracting image",f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
      magic = _read32(bytestream)
      if magic != 2051:
          raise ValueError("Invalid number %d in file %s",(magic,f.name))
      nums = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(nums * cols * rows)
      data = np.frombuffer(buf,dtype=np.uint8)
      data = data.reshape(nums,rows*cols)
  return data


def extract_labels(f):

  print("Extracting labels",f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
      magic = _read32(bytestream)
      if magic != 2049:
          raise ValueError("Invalid number %d in file %s",(magic,f.name))
      nums = _read32(bytestream)
      buf = bytestream.read(nums)
      data = np.frombuffer(buf,dtype=np.uint8)
      data = data.astype(int)
      return data

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {'pixes':features}
    ds = Dataset.from_tensor_slices((dict(features),targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds.shuffle(buffer_size=60000)
    features,labels = ds.make_one_shot_iterator().get_next()
    return features,labels

def construct_feature_columns():
    return [tf.feature_column.numeric_column("pixes",shape=784)]


if __name__ == '__main__':
    files = ["e:/data/train-images-idx3-ubyte.gz",
                 "e:/data/train-labels-idx1-ubyte.gz",
                 "e:/data/t10k-images-idx3-ubyte.gz",
                 "e:/data/t10k-labels-idx1-ubyte.gz"]

    with gfile.GFile(files[0],"rb") as f1, gfile.GFile(files[1],"rb") as f2:
        features = extract_images(f1)
        targets = extract_labels(f2)

    with gfile.GFile(files[2], "rb") as f1, gfile.GFile(files[3], "rb") as f2:
        predict_features = extract_images(f1)
        predict_targets = extract_labels(f2)
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.003)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5)
    my_classfier = tf.estimator.DNNClassifier(hidden_units=[3000,3000],
                                              n_classes=10,
                                              feature_columns=construct_feature_columns(),
                                              optimizer=my_optimizer,
                                              config=tf.estimator.RunConfig(keep_checkpoint_max=1),
                                              model_dir = 'E:/data/model3')


    train_input_fn = lambda:my_input_fn(features,targets,batch_size=35)
    predict_input_fn = lambda:my_input_fn(predict_features,predict_targets,num_epochs=1,shuffle=False)

    for loop in range(10):
        my_classfier.train(input_fn = train_input_fn,steps = 200)
        predictions = my_classfier.predict(input_fn = predict_input_fn)
        class_ids = np.array([item['class_ids'][0] for item in predictions])
        accuracy = metrics.accuracy_score(predict_targets,class_ids)
        print("accuracy %0.2f in %d" %(accuracy,loop))

    for file in glob.glob(os.path.join(my_classfier.model_dir, 'events.out*')):
        os.remove(file)


