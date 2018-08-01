## sample run

import tensorflow as tf 
import os 

from model import Model 
from six.moves import cPickle

save_dir = 'save'
number_of_characters = 500
prime = u''


with open(os.path.join(save_dir, 'config.pkl'),'rb') as f:
	saved_args = cPickle.load(f)
with open(os.path.join(save_dir, 'chars_vocab.pkl'),'rb') as f:
	chars, vocab = cPickle.load(f)

model = Model(saved_args, training = False)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())
	ckpt = tf.train.get_checkpoint_state(save_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		print(model.sample(sess, chars, vocab, number_of_characters,prime).encode('utf-8'))
