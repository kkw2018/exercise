import tensorflow as tf
import numpy as np
import os
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import isfile, join
import re
from tensorflow.python.data import Dataset
import pandas as pd

glove="E:/data/glove.6B/glove.6B.50d.txt"
emb_size=50
maxSeqLength=250
posfile="E:/data/aclImdb_v1/aclImdb/train/pos/"
negfile="E:/data/aclImdb_v1/aclImdb/train/neg/"
def loadglove(glove):
    vocab = []
    embd = []
    vocab.append('unk')
    embd.append([0]*emb_size)
    file = open(glove,'r',encoding='utf-8')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    np.save("vocab",vocab)
    np.save("embd",np.asarray(embd, dtype="float32"))


def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def processFile(posfile,negfile):
    positiveFiles = [posfile + f for f
                     in listdir(posfile) if isfile(join(posfile, f))]
    negativeFiles = [negfile + f for f
                     in listdir(negfile) if isfile(join(negfile, f))]

    numFiles=len(positiveFiles) + len(negativeFiles)

    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    lens = np.zeros(numFiles, dtype='int32')
    fileCounter = 0
    for pf in positiveFiles:
       with open(pf, "r",encoding='utf-8') as f:
           indexCounter = 0
           line=f.readline()
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = vocab.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 0 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           lens[fileCounter] = indexCounter
           fileCounter = fileCounter + 1


    for nf in negativeFiles:
       with open(nf, "r",encoding='utf-8') as f:
           indexCounter = 0
           line=f.readline()
           cleanedLine = cleanSentences(line)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = vocab.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 0 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           lens[fileCounter] = indexCounter
           fileCounter = fileCounter + 1
    np.save('idsMatrix', ids)
    np.save('lens', lens)

def getlabels(posfile,negfile):
    positiveFiles = [posfile + f for f
                     in listdir(posfile) if isfile(join(posfile, f))]
    negativeFiles = [negfile + f for f
                     in listdir(negfile) if isfile(join(negfile, f))]
    labels = [[1, 0] for row in range(len(positiveFiles))]
    labels.append([[0, 1] for row in range(len(negativeFiles))])
    np.save('labels', labels)


if not isfile("vocab.npy"):
    loadglove(glove)
vocab = np.load("vocab.npy").tolist()
embd = np.load("embd.npy")
if not isfile("idsMatrix.npy"):
    processFile(posfile, negfile)
ids = np.load("idsMatrix.npy")
lens = np.load("lens.npy")
if not isfile("labels.npy"):
    getlabels(posfile, negfile)
labels = np.load("labels.npy")

features = {'ids':ids,'lens':lens}

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds.shuffle(buffer_size=60000)
    features,labels = ds.make_one_shot_iterator().get_next()
    print(features,labels)
    return features,labels
my_input_fn(features,labels)

