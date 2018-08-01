import tensorflow as tf 
import numpy as np 
import os
import argparse
import time

from six.moves import cPickle
from utility import Textloader
from model import Model

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

save_data_dirhead = 'example1'
data_dir = 'data'
save_dir = 'save'

args.data_dir_full = data_dir_full = save_data_dirhead + '\\' + data_dir
args.save_dir_full = save_dir_full = save_data_dirhead + '\\' + save_dir

## RNN parameter
args.rnn_size = rnn_size = 128
args.num_layers = num_layers = 3

args.batch_size = batch_size = 50
args.seq_length = seq_length = 50
args.num_epochs = num_epochs = 50
args.learning_rate = learning_rate = 0.002
args.decay_rate = decay_rate = 0.95
args.morpheme_weight = morpheme_weight = 9

data_loader = Textloader(data_dir_full,batch_size,seq_length)
args.vocab_size = vocab_size = data_loader.vocab_size
args.vocab_mor_size = vocab_mor_size = data_loader.vocab_mor_size

chars_type = data_loader.chars_type #chars_type[i] = the morpheme of chars[i]

# save config file and char/vocab file
if not os.path.isdir(save_dir_full):
	os.makedirs(save_dir_full)
with open(os.path.join(save_dir_full,'config.pkl'),'wb') as f:
	cPickle.dump(args,f)	
with open(os.path.join(save_dir_full,'chars_vocab.pkl'),'wb') as f:
	cPickle.dump((data_loader.chars,data_loader.vocab),f)
with open(os.path.join(save_dir_full,'chars_vocab_mor.pkl'),'wb') as f:
	cPickle.dump((data_loader.chars_mor,data_loader.vocab_mor),f)
with open(os.path.join(save_dir_full,'chars_type.pkl'),'wb') as f:
	cPickle.dump((data_loader.chars_type),f)
	
# loss value save file
loss_file = open(os.path.join(save_dir_full,'loss_trajectory.txt'),'w')
time_file = open(os.path.join(save_dir_full,'time.txt'),'w')

# Model design
model = Model(args)
count = 0
#training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())

	for e in range(num_epochs):
		data_loader.reset_batch_pointer()
		sess.run(tf.assign(model.lr,learning_rate * (decay_rate ** e)))

		for b in range(data_loader.num_batches):
			start_time = time.time()
			x1,x2,y1,y2 = data_loader.next_batch()
			_,l = sess.run([model.train_op, model.mean_loss],feed_dict = {model.input_data_word : x1, model.input_data_mor : x2, model.target_word : y1, model.target_mor : y2, model.chars_type : chars_type})
			count += 1

			loss_file.write('%04d\t%f\n' % (count,l))
			time_file.write('%04d\t%f\n' % (count,time.time()-start_time))
			print('%02d/%02d : %02d/%02d - loss : %f' %(e+1,num_epochs,b+1,data_loader.num_batches,l))

			if b == data_loader.num_batches-1:
				checkpoint_path = os.path.join(save_dir_full,'model.ckpt')
				saver.save(sess,checkpoint_path)

loss_file.close()			