THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python train_ass_mask.py -wordfile glove-vgg-6.ridge -train word_association.txt -margin 1 -clip 1 -samplingtype MAX -epochs 5 -eta 0.05 -learner adagrad -updatewords 0 -save 1 -dim 428 -outfile result/modelname

python gen_emb_mask.py result/modelname.pickle result/modelname.txt

THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python train_ass_lexmask.py -wordfile glove-vgg-6.ridge -train word_association.txt -margin 1 -clip 1 -samplingtype MAX -epochs 5 -eta 0.005 -learner adagrad -updatewords 0 -save 1 -dim 428 -outfile result/modelname

python gen_emb_lexmask.py result/modelname.pickle result/modelname.txt

