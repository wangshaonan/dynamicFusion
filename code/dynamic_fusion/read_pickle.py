import cPickle as pickle
import sys

data = pickle.load(open(sys.argv[1]))
print data[0].get_value().shape
