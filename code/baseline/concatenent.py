import numpy as np
from sklearn.preprocessing import normalize
import sys

vocab0 = {}
vocab1 = {}
for line in open(sys.argv[1]):
    line = line.strip().split()
    vocab0[line[0]] = np.array([float(i) for i in line[1:]])
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
vec0 = np.array(vec0)
vec1 = np.array(vec1)

vec0 = vec0/np.linalg.norm(vec0,axis=1)[:,None]
vec1 = vec1/np.linalg.norm(vec1,axis=1)[:,None]

#full
outfile = open(sys.argv[3], 'w')
for num,v in enumerate(vocab):
    outfile.write(v+' '+' '.join([str(i) for i in vec0[num]])+' '+' '.join([str(i) for i in vec1[num]])+'\n')
