import tensorflow as tf
import re
from os import listdir
from os.path import isfile, join
import numpy as np
import lstm_input



# vocab_size = 2000
# embedding_size = 50
# model_dir = "E:/data"
# keep_prob = 1.0
# num_layers = 3
# pos_location='E:/data/aclImdb_v1/aclImdb/test/pos/'
# neg_location='E:/data/aclImdb_v1/aclImdb/test/neg/'
# maxSeqLength = 250
# batchSize = 24
#
#
#
#
# positiveFiles = [pos_location + f for f in listdir(pos_location) if isfile(join(pos_location, f))]
# negativeFiles = [neg_location + f for f in listdir(neg_location) if isfile(join(neg_location, f))]
# numWords = []
# for pf in positiveFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Positive files finished')
#
# for nf in negativeFiles:
#     with open(nf, "r", encoding='utf-8') as f:
#         line=f.readline()
#         counter = len(line.split())
#         numWords.append(counter)
# print('Negative files finished')
#
# numFiles = len(numWords)
# print('The total number of files is', numFiles)
# print('The total number of words in the files is', sum(numWords))
# print('The average number of words in the files is', sum(numWords)/len(numWords))
#
#
# strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
#
# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, "", string.lower())
#
#
#
# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#
# for nf in negativeFiles:
#    with open(nf, "r") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#
# np.save('idsMatrix', ids)





def lstm_model_fn(features, labels, mode,params):
    # [batch_size x sentence_size x embedding_size]
    inputs = tf.contrib.layers.embed_sequence(
        features['seqs'], vocab_size, embedding_size,
        initializer=tf.random_uniform_initializer(-1.0, 1.0))

    # create an LSTM cell of size 100
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(100,forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    # create the complete LSTM
    outputs, final_states = tf.nn.dynamic_rnn(
        lstm_cell, inputs, sequence_length=features['len'], dtype=tf.float32)

    # get the final hidden states of dimensionality [batch_size x sentence_size]
    outputs = final_states.h

    logits = tf.layers.dense(inputs=outputs, units=1)
    predictions = {
        "probabilities": tf.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sigmoid_cross_entropy(labels=labels,predictions=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions = 1 if predictions["probabilities"] > 0.5 else 0)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

#
# lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn,
#                                          model_dir=os.path.join(model_dir, 'lstm'))


