# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
import unicode_converter as uc 

class Model():
	def __init__(self,args,training=True):
		self.args = args
		if not training:
			args.batch_size = 1
			args.seq_length = 1

		# cell design 
		cells_12 = []
		cells_13 = []
		cells_23 = []

		for _ in range(args.num_layers):
			with tf.variable_scope("RNN12"):
				cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)
				cells_12.append(cell)
			with tf.variable_scope("RNN13"):
				cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)	
				cells_13.append(cell)
			with tf.variable_scope("RNN23"):
				cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)	
				cells_23.append(cell)				
		# using LSTM, there is another option(GRU)
		# bi-directional RNN is also an alternative	
		
		with tf.variable_scope("RNN12"):
			self.cell12 = cell12 = tf.contrib.rnn.MultiRNNCell(cells_12,state_is_tuple = True)
		with tf.variable_scope("RNN13"):	
			self.cell13 = cell13 = tf.contrib.rnn.MultiRNNCell(cells_13,state_is_tuple = True)
		with tf.variable_scope("RNN23"):	
			self.cell23 = cell23 = tf.contrib.rnn.MultiRNNCell(cells_23,state_is_tuple = True)


		# data input		
		input_data_1 = self.input_data_1 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		input_data_2 = self.input_data_2 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		input_data_3 = self.input_data_3 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_1 = self.target_1 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_2 = self.target_2 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_3 = self.target_3 = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		
		self.initial_state12 = initial_state12 = cell12.zero_state(args.batch_size,tf.float32)
		self.initial_state13 = initial_state13 = cell13.zero_state(args.batch_size,tf.float32)
		self.initial_state23 = initial_state23 = cell23.zero_state(args.batch_size,tf.float32)

		# one-hot
		input_onehot_1 = tf.one_hot(input_data_1,args.vocab1_size)
		input_onehot_2 = tf.one_hot(input_data_2,args.vocab2_size)
		input_onehot_3 = tf.one_hot(input_data_3,args.vocab3_size)
		# this part may be modified, but I have no idea.

		#====================#
		## softmax variable ##
		#====================#
		# for input 1X2
		input_w12 = tf.get_variable("input_RNN12_w",[args.vocab1_size+args.vocab2_size,args.rnn_input_size])
		input_b12 = tf.get_variable("input_RNN12_b",[args.rnn_input_size])

		# for input 1X3
		input_w13 = tf.get_variable("input_RNN13_w",[args.vocab1_size+args.vocab3_size,args.rnn_input_size])
		input_b13 = tf.get_variable("input_RNN13_b",[args.rnn_input_size])

		# for input 2X3
		input_w23 = tf.get_variable("input_RNN23_w",[args.vocab2_size+args.vocab3_size,args.rnn_input_size])
		input_b23 = tf.get_variable("input_RNN23_b",[args.rnn_input_size])

		# for output  12 X 13 -> 1
		output_w1 = tf.get_variable("output_1_w",[args.rnn_size*2,args.vocab1_size])
		output_b1 = tf.get_variable("output_1_b",[args.vocab1_size])

		# for output  12 X 23 -> 2
		output_w2 = tf.get_variable("output_2_w",[args.rnn_size*2,args.vocab2_size])
		output_b2 = tf.get_variable("output_2_b",[args.vocab2_size])

		# for output  13 X 23 -> 3
		output_w3 = tf.get_variable("output_3_w",[args.rnn_size*2,args.vocab3_size])
		output_b3 = tf.get_variable("output_3_b",[args.vocab3_size])

		#########################################
		# design diagram						#
		#########################################

		#==============#
		## RNN design ##
		#==============#
		# reshape input
		reshape_input1_onehot = tf.reshape(input_onehot_1,[args.batch_size*args.seq_length,args.vocab1_size])
		reshape_input2_onehot = tf.reshape(input_onehot_2,[args.batch_size*args.seq_length,args.vocab2_size])
		reshape_input3_onehot = tf.reshape(input_onehot_3,[args.batch_size*args.seq_length,args.vocab3_size])

		# concatenate input
		concat_input_12 = tf.concat([reshape_input1_onehot,reshape_input2_onehot],1)
		concat_input_13 = tf.concat([reshape_input1_onehot,reshape_input3_onehot],1)
		concat_input_23 = tf.concat([reshape_input2_onehot,reshape_input3_onehot],1)

		# first layer (input concat layer)
		input_rnn12 = tf.reshape(tf.matmul(concat_input_12,input_w12)+input_b12,[args.batch_size,args.seq_length,args.rnn_input_size])
		input_rnn13 = tf.reshape(tf.matmul(concat_input_13,input_w13)+input_b13,[args.batch_size,args.seq_length,args.rnn_input_size])
		input_rnn23 = tf.reshape(tf.matmul(concat_input_23,input_w23)+input_b23,[args.batch_size,args.seq_length,args.rnn_input_size])

		# second layer (RNN layer)
		outputs_12, last_state_12 = tf.nn.dynamic_rnn(cell12,input_rnn12,initial_state = initial_state12,dtype = tf.float32,scope = 'RNN12')
		outputs_13, last_state_13 = tf.nn.dynamic_rnn(cell13,input_rnn13,initial_state = initial_state13,dtype = tf.float32,scope = 'RNN13')
		outputs_23, last_state_23 = tf.nn.dynamic_rnn(cell23,input_rnn23,initial_state = initial_state23,dtype = tf.float32,scope = 'RNN23')

		# reshape RNN output
		outputs_12 = tf.reshape(outputs_12,[-1,args.rnn_size])
		outputs_13 = tf.reshape(outputs_13,[-1,args.rnn_size])
		outputs_23 = tf.reshape(outputs_23,[-1,args.rnn_size])

		# concatenate RNN outputs
		concat_output_1 = tf.concat([outputs_12,outputs_13],1)
		concat_output_2 = tf.concat([outputs_12,outputs_23],1)
		concat_output_3 = tf.concat([outputs_13,outputs_23],1)

		# third layer(RNN softmax layer)
		outputs_1 = tf.matmul(concat_output_1,output_w1)+output_b1
		outputs_2 = tf.matmul(concat_output_2,output_w2)+output_b2
		outputs_3 = tf.matmul(concat_output_3,output_w3)+output_b3

		#=================#
		## loss function ##
		#=================#

		self.probs_1 = tf.nn.softmax(outputs_1)
		self.probs_2 = tf.nn.softmax(outputs_2)
		self.probs_3 = tf.nn.softmax(outputs_3)
		self.final_state12 = last_state_12
		self.final_state13 = last_state_13
		self.final_state23 = last_state_23

		weights_1 = tf.ones([args.batch_size,args.seq_length])
		weights_2 = tf.ones([args.batch_size,args.seq_length])
		weights_3 = tf.ones([args.batch_size,args.seq_length])

		## we need to modify the loss function.
		loss_1 = tf.contrib.seq2seq.sequence_loss(logits = outputs_1, targets = target_1, weights = weights_1)
		loss_2 = tf.contrib.seq2seq.sequence_loss(logits = outputs_2, targets = target_2, weights = weights_2)
		loss_3 = tf.contrib.seq2seq.sequence_loss(logits = outputs_3, targets = target_3, weights = weights_3)
		
		self.mean_loss = tf.reduce_mean(loos_1)+tf.reduce_mean(loos_2)+tf.reduce_mean(loos_3)

		# set train rate 
		self.lr = tf.Variable(0.0, trainable = False)
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.mean_loss)

	# ==================== #
	# === writing part === #
	# ==================== #
	def sample(self,sess,first_arr,second_arr,third_arr,vocab1,vocab2,vocab3,vocab_mor,num=200,prime='나는 ', sampling_type = 1):
		state12 = sess.run(self.cell12.zero_state(1,tf.float32))
		state13 = sess.run(self.cell13.zero_state(1,tf.float32))
		state23 = sess.run(self.cell23.zero_state(1,tf.float32))

		temp_tensor = []
		for i in range(len(prime)):
			if data[i] == '':
				continue
			else:
				char1,char2,char3 = uc.chr_diss(data[i])	
				temp_tensor.append((vocab_1[char1],vocab_2[char2],vocab_3[char3]))				
			
		for i in range(len(temp_tensor)-1):
			x1 = np.zeros((1,1))
			x1[0,0] = temp_tensor[i][0]
			x2 = np.zeros((1,1))
			x2[0,0] = temp_tensor[i][1]
			x3 = np.zeros((1,1))
			x3[0,0] = temp_tensor[i][2]

			feed = {self.input_data_1 : x1, self.input_data_2 : x2, self.input_data_3 : x3, self.initial_state12 : state12, self.initial_state13 : state13, self.initial_state23 : state23}
			[state12, state13, state23] = sess.run([self.final_state12,self.final_state13,self.final_state23],feed)

		# 수정(?)	
		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t,np.random.rand(1)*s)))	


		ret = prime

		x1 = temp_tensor[-1][0]
		x2 = temp_tensor[-1][1]
		x3 = temp_tensor[-1][2]
		count = 0
		for _ in range(num):
			x1 = np.zeros((1,1))
			x2 = np.zeros((1,1))
			x3 = np.zeros((1,1))
			x1[0,0] = x1
			x2[0,0] = x2
			x3[0,0] = x3

			feed = {self.input_data_1 : x1, self.input_data_2 : x2, self.input_data_3 : x3, self.initial_state12 : state12, self.initial_state13 : state13, self.initial_state23 : state23}
			[probs_1,probs_2,probs_3,state12,state13,state23] = sess.run([self.probs_1,self.probs_2,self.probs_3,self.final_state2,self.final_state13,self.final_state23],feed)

			p1 = probs_1[0]
			p2 = probs_2[0]
			p3 = probs_3[0]

			sample_1 = weighted_pick(p1)
			sample_2 = weighted_pick(p2)
			sample_3 = weighted_pick(p3)
			
			count += 1
			if count % 100 == 0:
				print('%d/%d' % (count,num))

			temp_component = []
			temp_component.append(vocab1[sampe_1])
			temp_component.append(vocab2[sampe_2])
			temp_component.append(vocab3[sampe_3])	

			ret += uc.char_ass(temp_component)

		return ret