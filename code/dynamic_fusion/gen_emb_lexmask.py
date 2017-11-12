import sys
import cPickle as pickle
import numpy as np

words = []
for line in open('glove-vgg.ridge'):
    words.append(line.strip().split()[0])

data = pickle.load(open(sys.argv[1]))
mask = np.tanh(np.dot(data[0].get_value(), data[1].get_value())+data[2].get_value())
oo = open('lexmask.txt', 'w')
for ind in range(len(words)):
    oo.write(words[ind]+' '+' '.join([str(i) for i in mask[ind]])+'\n')
    #print words[ind], mask[ind]

emb =  data[0].get_value()*mask

outfile = open(sys.argv[2], 'w')
for ind,w in enumerate(emb):
    outfile.write(words[ind]+' '+' '.join([str(i) for i in w])+'\n')
