## sample run

import tensorflow as tf 
import os

from model import Model
from six.moves import cPickle

save_data_dirhead = 'example1'
save_dir = 'save'
number_of_characters = 500
prime = '나'

save_dir_full = save_data_dirhead + '\\' + save_dir

with open(os.path.join(save_dir_full, 'config.pkl'),'rb') as f:
	saved_args = cPickle.load(f)
with open(os.path.join(save_dir_full,'chars_vocab.pkl'),'rb') as f:
	chars,vocab = cPickle.load(f)
with open(os.path.join(save_dir_full,'chars_vocab_mor.pkl'),'rb') as f:	
	chars_mor,vocab_mor = cPickle.load(f)	

result_file = open(os.path.join(save_dir_full,'output.txt'),'w')

model = Model(saved_args,training=False)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())
	ckpt = tf.train.get_checkpoint_state(save_dir_full)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess,ckpt.model_checkpoint_path)
		result,mor_list = model.sample(sess,chars,vocab,chars_mor,vocab_mor,number_of_characters,prime)
		result_file.write('%s' % result)


result_file.close()