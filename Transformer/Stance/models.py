import warnings
import os, sys, re, io, nltk, torch
import pandas as pd

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(context="talk")

import itertools
import ast
from collections import Counter
from gensim import utils, matutils 
from gensim.corpora.dictionary import Dictionary
from gensim.utils import deprecated
from nltk.corpus import stopwords
from numpy import linalg as LA
from numpy.random import binomial
from numpy import dot, float32 as REAL, memmap as np_memmap, \
    double, array, zeros, vstack, sqrt, newaxis, integer, \
    ndarray, sum as np_sum, prod, argmax
from numpy.linalg import norm

from scipy import stats
from scipy.stats import bernoulli
from scipy.stats import ortho_group  # Requires version 0.18 of scipy
from scipy.spatial import distance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from sklearn.preprocessing import Normalizer, normalize
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, recall_score, precision_score

from six import string_types, integer_types
from six.moves import zip, range
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import ast, time
from tqdm import tqdm, trange
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
import re
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import *

stemmer_sn = SnowballStemmer("english")
stemmer = PorterStemmer()

stoplist = stopwords.words("english")
lemmatizer=WordNetLemmatizer()

import keras
from keras.callbacks import Callback,ModelCheckpoint, ReduceLROnPlateau    
from keras import backend as K
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Embedding
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Input
from keras.models import Model
from keras.layers import LSTM,GRU,Dense
from keras.utils import Sequence,to_categorical


def focal_loss(gamma=2., weights=1):   #weights np.asarray()
    weights= K.variable(weights)
    def focal_loss_fixed(y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(),1)
        y_pred = K.clip(y_pred,K.epsilon(),1)
        return - K.sum(weights* K.pow(1. - y_pred, gamma)* y_true * K.log(y_pred), axis=-1) 
    return focal_loss_fixed

class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, batch, logs={}):
        prob = self.model.predict(self.validation_data[0])
        predict = np.squeeze(prob>=0.5)*1
        targ = np.squeeze(self.validation_data[1])
        f1s = f1_score(targ, predict, average='macro')
        self.val_f1s.append(f1s)
        #print(" - val_f1: %f " %(f1s))
        return

def plot_confusion_matrix(cm, target_names, title='Confusion matrix (f1-score)',cmap=None, normalize=True):
    
    import itertools
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    
    
def train_model(model, x_train, y_train, x_val, y_val, class_weights, e=20, BS=32, verb=1, focal=False):
    if focal:
        model.compile(loss=focal_loss(2,weights=class_weights),optimizer='adam',metrics=['acc'])
    else:
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    
    calls = []
    metrics = Metrics()
    calls.append(metrics)
    
    if not focal:
        hist=model.fit(x_train, y_train, batch_size=BS, epochs=e, verbose=verb, validation_data=(x_val,y_val), class_weight=class_weights, callbacks=calls)
    else:
        hist=model.fit(x_train, y_train, batch_size=BS, epochs=e, verbose=verb, validation_data=(x_val,y_val), callbacks=calls)
  
    return model, hist 


def create_cnn1(shape):
    sequence_input = Input(shape=shape)
    batch = BatchNormalization()(sequence_input)
    cov1= Conv1D(128, 5, activation='relu',padding='same')(batch)
    pool = MaxPooling1D(pool_size=3)(cov1)    
    batch = BatchNormalization()(pool)
    drop = Dropout(0.5)(batch)
    cov1= Conv1D(64, 5, activation='relu',padding='same')(drop)
    pool = MaxPooling1D(pool_size=3)(cov1)
    batch = BatchNormalization()(pool)
    drop = Dropout(0.5)(batch)
    flat = Flatten()(drop)
    preds = Dense(100, activation='relu')(flat)
    preds = BatchNormalization()(preds)
    preds = Dense(4, activation='softmax')(preds)
    cnn1 = Model(sequence_input, preds)
    return cnn1

def create_cnn2(shape):
    sequence_input = Input(shape=shape)
    batch = BatchNormalization()(sequence_input)
    cov1= Conv1D(128, 5, activation='relu',padding='same')(batch)
    pool = MaxPooling1D(pool_size=3)(cov1)    
    batch = BatchNormalization()(pool)
    drop = Dropout(0.5)(batch)
    cov1= Conv1D(64, 5, activation='relu',padding='same')(drop)
    pool = MaxPooling1D(pool_size=3)(cov1)
    batch = BatchNormalization()(pool)
    drop = Dropout(0.5)(batch)
    flat = Flatten()(drop)
    preds = Dense(128, activation='relu')(flat)
    preds = BatchNormalization()(preds)
    preds= Dropout(0.3)(preds)
    preds = Dense(4, activation='softmax')(preds)
    cnn2 = Model(sequence_input, preds)
    return cnn2


from keras.utils import Sequence,to_categorical
from keras.layers import CuDNNGRU,CuDNNLSTM

def create_complex_GRU_2(unidades,unidades2,opt,input_s):
    model = Sequential()
    model.add(CuDNNGRU(units=unidades,return_sequences=True,input_shape=input_s))
    model.add(Dropout(0.45))
    model.add(BatchNormalization())
    model.add(CuDNNGRU(units=unidades2,return_sequences=False,input_shape=input_s ))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax')) 
    return model

def create_complex_GRU_3(unidades,unidades2,unidades3,opt,input_s):
    model = Sequential()
    model.add(CuDNNGRU(units=unidades,return_sequences=True,input_shape=input_s))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(CuDNNGRU(units=unidades2,return_sequences=True,input_shape=input_s ))
    model.add(Dropout(0.45))
    model.add(BatchNormalization())
    model.add(CuDNNGRU(units=unidades3,return_sequences=False,input_shape=input_s ))#,
    model.add(Dropout(0.35))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='softmax')) # 
    return model

