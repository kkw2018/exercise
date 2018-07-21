import tensorflow as tf
import numpy as np
import os
from os import listdir
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os.path import isfile, join
import re
import codecs
glove="C:/baiduyun/glove.6B.50d.txt"
emb_size=50
maxSeqLength=250
posfile="C:/baiduyun/aclImdb/train/pos/"
negfile="C:/baiduyun/aclImdb/train/neg/"
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
           fileCounter = fileCounter + 1
    np.save('idsMatrix', ids)

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
if not isfile("labels.npy"):
    getlabels(posfile, negfile)
labels = np.load("labels.npy")

