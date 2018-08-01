import tensorflow as tf 
import numpy as np 
import os
import argparse

from six.moves import cPickle
from utility import TextLoader
from model import Model

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

args.data_dir = data_dir = 'data' # input text directory
args.save_dir = save_dir = 'save' # store checkpointed model directory

args.rnn_size = rnn_size = 128
args.num_layers = num_layers = 2

args.batch_size = batch_size = 50
args.seq_length = seq_length = 50
args.num_epochs = num_epochs = 50
args.learning_rate = learning_rate = 0.002
args.decay_rate = decay_rate = 0.97


data_loader = TextLoader(data_dir,batch_size,seq_length)
args.vocab_size = vocab_size = data_loader.vocab_size

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
with open(os.path.join(save_dir,'config.pkl'),'wb') as f:
	cPickle.dump(args, f)
with open(os.path.join(save_dir,'chars_vocab.pkl'),'wb') as f:
	cPickle.dump((data_loader.chars, data_loader.vocab), f)	


## Model design
model = Model(args)

## training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())

	for e in range(num_epochs):
		data_loader.reset_batch_pointer()
		sess.run(tf.assign(model.lr,learning_rate * (decay_rate ** e)))

		for b in range(data_loader.num_batches):
			x,y = data_loader.next_batch()
			_,l = sess.run([model.train_op, model.mean_loss],
				feed_dict={model.input_data : x, model.targets : y})

			print('%d/%d : %d/%d - loss : %f' % (e+1,num_epochs,b+1,data_loader.num_batches,l))
			if b == data_loader.num_batches-1:
				checkpoint_path = os.path.join(save_dir,'model.ckpt')
				saver.save(sess,checkpoint_path)












