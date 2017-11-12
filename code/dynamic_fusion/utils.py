import lasagne
import theano
import random
import numpy as np
import argparse
import sys
from theano import tensor as T
from theano import config
import cPickle
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from random import randint
from random import choice
from scipy.stats import spearmanr
from scipy.stats import pearsonr

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    else:
	print w
	print 'oov'
	sys.exit()
        #return words['UUUNKKK']

class tree(object):

    def __init__(self, phrase, words):
        self.phrase = phrase
        self.embeddings = []
        self.representation = None

    def populate_embeddings(self, words):
        phrase = self.phrase.lower()
        arr = phrase.split()
        for i in arr:
            self.embeddings.append(lookupIDX(words,i))

    def unpopulate_embeddings(self):
        self.embeddings = []

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def getData(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split()
            if len(i) == 3:
                e = (tree(i[0], words), tree(i[1], words), i[2])
                examples.append(e)
            else:
                print i
    return examples

def checkIfQuarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False

def saveParams(model, fname):
    f = file(fname, 'wb')
    cPickle.dump(model.all_params, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def prepare_data(list_of_seqs):
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s

    return x


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)

def getPairRand(d,idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def getPairMixScore(d,idx,maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return getPairRand(d,idx)

def getPairsFast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i][0:2]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        (t1,t2) = d[i][0:2]
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = getPairRand(d,i)
            p2 = getPairRand(d,i)
        if type == "MIX":
            p1 = getPairMixScore(d,i,T[arr[2*i]])
            p2 = getPairMixScore(d,i,T[arr[2*i+1]])
        pairs.append((p1,p2))
    return pairs

def getpairs(model, batch, params):
    g1 = []
    g2 = []

    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x = prepare_data(g1)
    g2x = prepare_data(g2)

    embg1 = model.feedforward_function(g1x)
    embg2 = model.feedforward_function(g2x)

    for idx, i in enumerate(batch):
        i[0].representation = embg1[idx, :]
        i[1].representation = embg2[idx, :]

    pairs = getPairsFast(batch, params.type)
    p1 = []
    p2 = []
    for i in pairs:
        p1.append(i[0].embeddings)
        p2.append(i[1].embeddings)

    p1x = prepare_data(p1)
    p2x = prepare_data(p2)

    return (g1x,  g2x,  p1x,  p2x)


def getSeqs(p1,p2,words):
    p1 = p1.split()
    p2 = p2.split()
    X1 = []
    X2 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    for i in p2:
        X2.append(lookupIDX(words,i))
    return X1, X2

def getSeq(p1,words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1

def getCorrelation(model,words,test):
    preds = []
    golds = []
    seq1 = []
    seq2 = []
    for i in test:
        p1 = i[0]; p2 = i[1]; score = float(i[2])
        X1, X2 = getSeqs(p1.phrase,p2.phrase,words)
        seq1.append(X1)
        seq2.append(X2)
        golds.append(score)
    x1 = prepare_data(seq1)
    x2 = prepare_data(seq2)
    scores = model.scoring_function(x1,x2)
    preds = np.squeeze(scores)
    return pearsonr(preds,golds)[0], spearmanr(preds,golds)[0]

def evaluate(model,words, test):

    p,s = getCorrelation(model,words,test)

    print str(p)+' '+str(s)
    return s
