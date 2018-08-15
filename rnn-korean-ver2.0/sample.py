## sample run

# -*- coding: utf-8 -*-
import tensorflow as tf 
import os

from model import Model
from six.moves import cPickle

save_data_dirhead = 'example1'
save_dir = 'save'
number_of_characters = 500
prime = '나는 '

save_dir_full = save_data_dirhead + '\\' + save_dir

with open(os.path.join(save_dir_full,'config.pkl'),'rb') as f:
	saved_args = cPickle.load(f)
with open(os.path.join(save_dir_full,'vocab1.pkl'),'rb') as f:
	first_arr,vocab1 = cPickle.load(f)
with open(os.path.join(save_dir_full,'vocab2.pkl'),'rb') as f:
	second_arr,vocab2 = cPickle.load(f)
with open(os.path.join(save_dir_full,'vocab3.pkl'),'rb') as f:
	third_arr,vocab3 = cPickle.load(f)
	
result_file = open(os.path.join(save_dir_full,'output.txt'),'w')

model = Model(saved_args,training=False)
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())
	ckpt = tf.train.get_checkpoint_state(save_dir_full)
	if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess,ckpt.model_checkpoint_path)
		result = model.sample(sess,first_arr,second_arr,third_arr,vocab1,vocab2,vocab3,number_of_characters,prime)
		result_file.write('%s' % result)
'''
result_detail_file = open(os.path.join(save_dir_full,'output_detail.txt'),'w')

for i in range(len(mor_list)):
	result_detail_file.write('%d of output - %s : %c\n' % (i+1,char_list[i],mor_list[i]))

result_detail_file.close()	
'''	
result_file.close()