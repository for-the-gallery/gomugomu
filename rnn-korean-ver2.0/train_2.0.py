# -*- coding: utf-8 -*-
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

args.data_dir_full = data_dir_full = save_data_dirhead + '/' + data_dir # mac directory format, if using window replace '/' to '\\'
args.save_dir_full = save_dir_full = save_data_dirhead + '/' + save_dir

## RNN parameter
args.rnn_size = rnn_size = 1024
args.num_layers = num_layers = 2
args.rnn_input_size = rnn_input_size = 1024

args.batch_size = batch_size = 50
args.seq_length = seq_length = 50
args.num_epochs = num_epochs = 50
args.learning_rate = learning_rate = 0.002
args.decay_rate = decay_rate = 0.95

## Data loading...
data_loader = Textloader(data_dir_full,batch_size,seq_length,encoding='utf-8')
args.vocab1_size = vocab1_size = data_loader.vocab1_size
args.vocab2_size = vocab2_size = data_loader.vocab2_size
args.vocab3_size = vocab3_size = data_loader.vocab3_size

# save config file and char/vocab file
if not os.path.isdir(save_dir_full):
	os.makedirs(save_dir_full)
with open(os.path.join(save_dir_full,'config.pkl'),'wb') as f:
	cPickle.dump(args,f)	
with open(os.path.join(save_dir_full,'vocab1.pkl'),'wb') as f:
	cPickle.dump((data_loader.first_arr,data_loader.vocab1),f)
with open(os.path.join(save_dir_full,'vocab2.pkl'),'wb') as f:
	cPickle.dump((data_loader.second_arr,data_loader.vocab2),f)
with open(os.path.join(save_dir_full,'vocab3.pkl'),'wb') as f:
	cPickle.dump((data_loader.third_arr,data_loader.vocab3),f)
	
# loss value and duration time save file
loss_file = open(os.path.join(save_dir_full,'loss_trajectory.txt'),'w')
time_file = open(os.path.join(save_dir_full,'time.txt'),'w')

# Model design
model = Model(args)
count = 0

# for decrease the learning rate if there is no enough change of loss value. 
l_prev = 0
l_present = 0

#training
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())

	for e in range(num_epochs):
		data_loader.reset_batch_pointer()
		sess.run(tf.assign(model.lr,learning_rate * (decay_rate ** e)))
		temp_l = []
		for b in range(data_loader.num_batches):
			start_time = time.time()
			x1,x2,x3,y1,y2,y3 = data_loader.next_batch()
			_,l = sess.run([model.train_op, model.mean_loss],feed_dict = {model.input_data_1 : x1, model.input_data_2 : x2, model.input_data_3 : x3, model.target_1 : y1, model.target_2 : y2, model.target_3 : y3})
			count += 1
			temp_l.append(l)

			loss_file.write('%04d\t%f\n' % (count,l))
			time_file.write('%04d\t%f\n' % (count,time.time()-start_time))
			print('%02d/%02d : %02d/%02d - loss : %f' %(e+1,num_epochs,b+1,data_loader.num_batches,l))

			if b == data_loader.num_batches-1:
				checkpoint_path = os.path.join(save_dir_full,'model.ckpt')
				saver.save(sess,checkpoint_path)

				# decreasing part of learning rate
				'''
				l_present = np.average(temp_l)
				if e == 0:
					l_prev = l_present
				else:
					if l_prev * 0.9 < l_present:
						learning_rate = learning_rate *0.5
						l_prev = l_present
					else:	
						l_prev = l_present
				'''


loss_file.close()
time_file.close()			