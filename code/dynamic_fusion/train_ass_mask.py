import lasagne
import theano
import random
import numpy as np
import argparse
from theano import tensor as T
from theano import config
import time
import utils
import sys


def str2bool(v):
    if v is None:
        return False
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise ValueError('A type that was supposed to be boolean is not boolean.')

def learner2bool(v):
    if v is None:
        return lasagne.updates.adagrad
    if v.lower() == "adagrad":
        return lasagne.updates.adagrad
    if v.lower() == "adam":
        return lasagne.updates.adam
    raise ValueError('A type that was supposed to be a learner is not.')

class lasagne_average_layer(lasagne.layers.MergeLayer):
    def __init__(self, incoming, num_units=428, W=lasagne.init.Constant(1.0), b=lasagne.init.Normal(), **kwargs):
        super(lasagne_average_layer, self).__init__(incoming, **kwargs)
	self.W = self.add_param(W, (428,))
	#self.b = self.add_param(b, (428,))

    def get_output_for(self, inputs, **kwargs):
        emb = inputs[0]
        emb = emb.sum(axis=1) * self.W[None,:]
        return emb

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

class word_ass_model(object):
    def __init__(self, We_initial,params):
        initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        #symbolic params
        g1batchindices = T.imatrix(); g2batchindices = T.imatrix()
        p1batchindices = T.imatrix(); p2batchindices = T.imatrix()

        #get embeddings
        l_in = lasagne.layers.InputLayer((None, None, 1))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0], output_size=We.get_value().shape[1], W=We)
        l_average = lasagne_average_layer([l_emb])

        embg1 = lasagne.layers.get_output(l_average, {l_in:g1batchindices})
        embg2 = lasagne.layers.get_output(l_average, {l_in:g2batchindices})
        embp1 = lasagne.layers.get_output(l_average, {l_in:p1batchindices})
        embp2 = lasagne.layers.get_output(l_average, {l_in:p2batchindices})

        #objective function
        g1g2 = (embg1*embg2).sum(axis=1)
        g1g2norm = T.sqrt(T.sum(embg1**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        g1g2 = g1g2 / g1g2norm

        p1g1 = (embp1*embg1).sum(axis=1)
        p1g1norm = T.sqrt(T.sum(embp1**2,axis=1)) * T.sqrt(T.sum(embg1**2,axis=1))
        p1g1 = p1g1 / p1g1norm

        p2g2 = (embp2*embg2).sum(axis=1)
        p2g2norm = T.sqrt(T.sum(embp2**2,axis=1)) * T.sqrt(T.sum(embg2**2,axis=1))
        p2g2 = p2g2 / p2g2norm

        costp1g1 = params.margin - g1g2 + p1g1
        costp1g1 = costp1g1*(costp1g1 > 0)

        costp2g2 = params.margin - g1g2 + p2g2
        costp2g2 = costp2g2*(costp2g2 > 0)

        cost = costp1g1 + costp2g2
	
        self.all_params = lasagne.layers.get_all_params(l_average, trainable=True)
	self.network_params = lasagne.layers.get_all_params(l_average, trainable=True)
        self.network_params.pop(0)

        word_reg = 0.5*params.LW*lasagne.regularization.l2(We-initial_We)
        cost = T.mean(cost) + word_reg

        #feedforward
        self.feedforward_function = theano.function([g1batchindices], embg1)
        self.cost_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices], cost)

        prediction = g1g2

        self.scoring_function = theano.function([g1batchindices, g2batchindices],prediction)
	
        #updates
	if params.updatewords:
            grads = theano.gradient.grad(cost, self.all_params)
            if params.clip:
                grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
            updates = params.learner(grads, self.all_params, params.eta)
	else:
	    grads = theano.gradient.grad(cost, self.network_params)
            if params.clip:
                grads = [lasagne.updates.norm_constraint(grad, params.clip, range(grad.ndim)) for grad in grads]
            updates = params.learner(grads, self.network_params, params.eta)


        self.train_function = theano.function([g1batchindices, g2batchindices, p1batchindices, p2batchindices], cost, updates=updates)

def train(model, data, words, params):
    start_time = time.time()
    counter = 0
    pre_ss = 0
    try:
        for eidx in xrange(params.epochs):
            #10 cross validation
            #nc = len(data)/10
            #test_nn = range((eidx%10)*nc, ((eidx%10)+1)*nc)
            #train_nn = list(set(xrange(len(data)))-set(test_nn))
            #test = [data[i] for i in test_nn]
            #data = [data[i] for i in train_nn]


            kf = utils.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:
                uidx += 1
                batch = [data[t] for t in train_index]
                for i in batch:
                    i[0].populate_embeddings(words)
                    i[1].populate_embeddings(words)

                (g1x, g2x, p1x, p2x) = utils.getpairs(model, batch, params)

                cost = model.train_function(g1x, g2x, p1x, p2x)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                #if (utils.checkIfQuarter(uidx, len(kf))):
                #    ss = utils.evaluate(model, words, test)
		#    if (params.save) and ss > pre_ss:
                #        counter += 1
                #        utils.saveParams(model, params.outfile+str(eidx) + '.pickle')
		#	pre_ss = ss
                #    sys.stdout.flush()

                # undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    i[0].unpopulate_embeddings()
                    i[1].unpopulate_embeddings()

                    # print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

            #ss = utils.evaluate(model, words, test)
	    #if (params.save) and ss > pre_ss:
            #    counter += 1
            #    utils.saveParams(model, params.outfile+str(eidx) + '.pickle')
	    #    pre_ss = ss
	    utils.saveParams(model, params.outfile+str(eidx)+'.pickle')

            print 'Epoch ', (eidx + 1), 'Cost ', cost

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    print "total time:", (end_time - start_time)

class params(object):
    def __init__(self):
        self.LW = 1e-05
        self.LC = 1e-05
        self.eta = 0.5

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    
    params = params()
    parser = argparse.ArgumentParser()
    parser.add_argument("-LW", help="Lambda for word embeddings (normal training).", default=1e-05, type=float)
    parser.add_argument("-LC", help="Lambda for composition parameters (normal training).", default=1e-05, type=float)
    parser.add_argument("-outfile", help="Output file name.", default='outfile')
    parser.add_argument("-batchsize", help="Size of batch.", default=25, type=int)
    parser.add_argument("-dim", help="Size of input.", default=428, type=int)
    parser.add_argument("-wordfile", help="Word embedding file.", default='glove.test')
    parser.add_argument("-layersize", help="Size of output layers in models.", type=int)
    parser.add_argument("-updatewords", help="Whether to update the word embeddings",default='1')
    parser.add_argument("-wordstem", help="Nickname of word embeddings used.",default='wordstem')
    parser.add_argument("-save", help="Whether to pickle the model.",default='1')
    parser.add_argument("-train", help="Training data file.",default='word_association_pair.txt')
    parser.add_argument("-margin", help="Margin in objective function.", default=0.5, type=float)
    parser.add_argument("-clip", help="Threshold for gradient clipping.", default=1, type=int)
    parser.add_argument("-samplingtype", help="Type of sampling used.",default='MAX')
    parser.add_argument("-epochs", help="Number of epochs in training.", default=5, type=int)
    parser.add_argument("-eta", help="Learning rate.",default=0.5, type=float)
    parser.add_argument("-learner", help="Either AdaGrad or Adam.",default='AdaGrad')

    args = parser.parse_args()

    params.LW = args.LW
    params.LC = args.LC
    params.eta = args.eta
    params.outfile = args.outfile
    params.batchsize = args.batchsize
    params.hiddensize = args.dim
    params.wordfile = args.wordfile
    params.layersize = args.layersize
    params.updatewords = str2bool(args.updatewords)
    params.wordstem = args.wordstem
    params.save = str2bool(args.save)
    params.train = args.train
    params.margin = args.margin
    params.clip = args.clip
    params.type = args.samplingtype
    params.epochs = args.epochs
    params.learner = learner2bool(args.learner)


    (words, We) = utils.getWordmap(params.wordfile)
    examples = utils.getData(params.train, words)


    model = word_ass_model(We, params)

    train(model, examples, words, params)
