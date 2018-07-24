import tensorflow as tf
import re
from os import listdir
from os.path import isfile, join
import numpy as np
import lstm_input
from sklearn import metrics
import os
import glob

def lstm_cell_func(lstm_size):
  return tf.contrib.rnn.BasicLSTMCell(lstm_size)


def lstm_model_fn(features, labels, mode,params):
    # [batch_size x sentence_size x embedding_size]
    inputs = tf.nn.embedding_lookup(lstm_input.embd,features['seqs'])
    # create an LSTM cell of size 100
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_func(100) for _ in range(lstm_input.num_layers)])


    # create the complete LSTM
    outputs, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)

    # get the final hidden states of dimensionality [batch_size x sentence_size]
    outputs = tf.reshape(tf.concat(1, outputs), [-1, lstm_input.hidden_size])
    logits = tf.layers.dense(inputs=outputs, units=2,activation=tf.nn.relu)
    predicted_classes = tf.argmax(logits, 1)
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(indices=labels,depth=2),logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=lstm_input.learning_rate)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

lstm_classifier = tf.estimator.Estimator(
        model_fn=lstm_model_fn, model_dir=lstm_input.model_dir,
        params={'feature_columns': lstm_input.construct_feature_columns()
                }
)

train_input_fn = lambda: lstm_input.my_input_fn(lstm_input.features, lstm_input.labels, batch_size=1)
predict_input_fn = lambda: lstm_input.my_input_fn(lstm_input.features, lstm_input.labels, num_epochs=1, shuffle=False)

for loop in range(20):
    lstm_classifier.train(input_fn=train_input_fn, steps=100)
    predictions = lstm_classifier.predict(input_fn=predict_input_fn)
    class_ids = np.array([item['class_ids'][0] for item in predictions])
    accuracy = metrics.accuracy_score(lstm_input.labels, class_ids)
    print("accuracy %0.4f in %d" % (accuracy, loop))

for file in glob.glob(os.path.join(lstm_classifier.model_dir, 'events.out*')):
    os.remove(file)

