import sys
from sklearn.linear_model import Ridge
import numpy as np
from numpy import random

#vec0 is linguistic, vec1 is image
vocab0 = {}
vocab1 = {}
word_dim = 0
for line in open(sys.argv[1]):
    line = line.strip().split()
    vocab0[line[0]] = np.array([float(i) for i in line[1:]])
word_dim = len(line[1:])
for line in open(sys.argv[2]):
    line = line.strip().split()
    vocab1[line[0]] = np.array([float(i) for i in line[1:]])

vocab = set(vocab0.keys())&set(vocab1.keys())
vocab = list(vocab)

vec0 = []
vec1 = []
for v in vocab:
    vec0.append(vocab0[v])
    vec1.append(vocab1[v])

X = np.array(vec0) 
Y = np.array(vec1)

P = np.zeros((len(vocab0), word_dim))
for num,v in enumerate(vocab0):
    P[num] = vocab0[v]

print P.shape
#P = np.array(vec2)

para = sys.argv[5]
clf = Ridge(alpha=float(para))
clf.fit(X, Y)

oo = clf.predict(P)
#full-1

outfile1 = open(sys.argv[3], 'w')
outfile2 = open(sys.argv[4], 'w')
for num, v in enumerate(vocab0):
    lin = vocab0[v]/np.linalg.norm(vocab0[v])
    img = oo[num]/np.linalg.norm(oo[num])
    out = np.concatenate((lin, img), axis=0)
    outfile1.write(v+' '+' '.join([str(i) for i in out])+'\n')
    outfile2.write(v+' '+' '.join([str(i) for i in oo[num]])+'\n')
