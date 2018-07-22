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
import mnist_input

'''
1.download data
2.input func
3.feature columns
4.creat model_fn : define graph model,renturn train,predict,validate results
5.train,predict,validate
'''


def my_model_fn(features,labels,mode,params):
    net = tf.feature_column.input_layer(features,params['feature_columns'])
    net = tf.reshape(net,[-1,28,28,1])
    conv1 = tf.layers.conv2d(
        inputs=net,
        filters=32,
        kernel_size=[5,5],
        padding='same',
        activation=tf.nn.relu
    )
    pool1 = tf.layers.max_pooling2d(
        strides=2,
        inputs=conv1,
        pool_size=[2,2]
    )

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu
    )

    pool2 = tf.layers.max_pooling2d(
        strides=2,
        inputs=conv2,
        pool_size=[2, 2]
    )

    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat,units=1024,activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense,rate=0.6,training = mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout,units=10,activation=tf.nn.relu)
    predicted_classes = tf.argmax(logits, 1)
    predictions = {
      'class_ids': predicted_classes[:, tf.newaxis],
      "probabilities":tf.nn.softmax(logits,name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])
    }
    return  tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


files = ["e:/data/train-images-idx3-ubyte.gz",
         "e:/data/train-labels-idx1-ubyte.gz",
         "e:/data/t10k-images-idx3-ubyte.gz",
         "e:/data/t10k-labels-idx1-ubyte.gz"]

with gfile.GFile(files[0], "rb") as f1, gfile.GFile(files[1], "rb") as f2:
    features = mnist_input.extract_images(f1)
    targets = mnist_input.extract_labels(f2)
with gfile.GFile(files[2], "rb") as f1, gfile.GFile(files[3], "rb") as f2:
    predict_features = mnist_input.extract_images(f1)
    predict_targets = mnist_input.extract_labels(f2)

mnist_classifier = tf.estimator.Estimator(
        model_fn=my_model_fn, model_dir="E:/data/model5",
        params={'feature_columns': mnist_input.construct_feature_columns()
                }
)

train_input_fn = lambda: mnist_input.my_input_fn(features, targets, batch_size=100)
predict_input_fn = lambda: mnist_input.my_input_fn(predict_features, predict_targets, num_epochs=1, shuffle=False)

for loop in range(20):
    mnist_classifier.train(input_fn=train_input_fn, steps=200)
    predictions = mnist_classifier.predict(input_fn=predict_input_fn)
    class_ids = np.array([item['class_ids'][0] for item in predictions])
    accuracy = metrics.accuracy_score(predict_targets, class_ids)
    print("accuracy %0.4f in %d" % (accuracy, loop))

for file in glob.glob(os.path.join(mnist_classifier.model_dir, 'events.out*')):
    os.remove(file)