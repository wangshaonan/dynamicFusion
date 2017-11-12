import sys
import cPickle as pickle

words = []
for line in open('glove-vgg-full.emb'):
    words.append(line.strip().split()[0])

data = pickle.load(open(sys.argv[1]))
emb =  data[0].get_value()

outfile = open(sys.argv[2], 'w')
for ind,w in enumerate(emb):
    outfile.write(words[ind]+' '+' '.join([str(i) for i in w])+'\n')
