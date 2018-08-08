import tensorflow as tf 
import numpy as np 
from konlpy.tag import Kkma

class Model():
	def __init__(self,args,training=True):
		self.args = args
		if not training:
			args.batch_size = 1
			args.seq_length = 1

		# cell design 
		cells_mor = []
		cells_word = []
		for _ in range(args.num_layers):
			with tf.variable_scope("mor_LSTM"):
				cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)
				cells_mor.append(cell)
			with tf.variable_scope("word_LSTM"):
				cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)	
				cells_word.append(cell)
		# using LSTM, there is another option(GRU)
		# bi-directional RNN is also an alternative	
		
		with tf.variable_scope("LSTM1"):
			self.cell1 = cell1 = tf.contrib.rnn.MultiRNNCell(cells_mor,state_is_tuple = True)
		with tf.variable_scope("LSTM2"):	
			self.cell2 = cell2 = tf.contrib.rnn.MultiRNNCell(cells_word,state_is_tuple = True)

		# data input		
		input_data_word = self.input_data_word = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		input_data_mor = self.input_data_mor = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_word = self.target_word = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_mor = self.target_mor = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		chars_type = self.chars_type = tf.placeholder(tf.int32,[args.vocab_size]) # transform from word to mor

		self.initial_state1 = initial_state1 = cell1.zero_state(args.batch_size,tf.float32)
		self.initial_state2 = initial_state2 = cell2.zero_state(args.batch_size,tf.float32)

		# one-hot
		input_onehot_word = tf.one_hot(input_data_word,args.vocab_size)
		input_onehot_mor = tf.one_hot(input_data_mor,args.vocab_mor_size)
		# this part may be modified, but I have no idea.

		#====================#
		## softmax variable ##
		#====================#
		# for first input word * input morpheme
		softmax_w1 = tf.get_variable("input_first_RNN_w",[args.vocab_mor_size+args.vocab_size,args.rnn_input_size_1])
		softmax_b1 = tf.get_variable("input_first_RNN_b",[args.rnn_input_size_1])

		# for second input word * input morpheme
		softmax_w2 = tf.get_variable("input_second_RNN_w",[args.vocab_mor_size+args.vocab_size,args.rnn_input_size_2])
		softmax_b2 = tf.get_variable("input_second_RNN_b",[args.rnn_input_size_2])

		# for outputs to morpheme
		softmax_w3 = tf.get_variable("softmax_for_mor_w",[args.rnn_size+args.rnn_size,args.vocab_mor_size])
		softmax_b3 = tf.get_variable("softmax_for_mor_b",[args.vocab_mor_size])

		# for outputs to word
		softmax_w4 = tf.get_variable("softmax_for_input_w",[args.rnn_size+args.rnn_size,args.vocab_size])
		softmax_b4 = tf.get_variable("softmax_for_input_b",[args.vocab_size])


		#########################################
		# design diagram						#
		#										#
		#	   word outputs	  mor outputs		#
		#			|			|				#
		#			-------------				#
		#				  |						#
		#		 ---------X--------				#
		#		  |				  |				# 
		#		LSTM1			LSTM2			#
		#		  |				  |				#
		#         ----------------				#
		#				 |						#
		#		---------X-------				#
		#		|				|				#
		#	word inputs 	mor inputs   		#
		#										#
		#########################################

		#==============#
		## RNN design ##
		#==============#
		# concatenate two inputs
		reshape_input_onehot_word = tf.reshape(input_onehot_word,[args.batch_size*args.seq_length,args.vocab_size])
		reshape_input_onehot_mor = tf.reshape(input_onehot_mor,[args.batch_size*args.seq_length,args.vocab_mor_size])

		concat_input = tf.concat([reshape_input_onehot_word,reshape_input_onehot_mor],1)

		# inputs for RNN
		input_rnn1 = tf.reshape(tf.matmul(concat_input,softmax_w1)+softmax_b1,[args.batch_size,args.seq_length,args.rnn_input_size_1])
		input_rnn2 = tf.reshape(tf.matmul(concat_input,softmax_w2)+softmax_b2,[args.batch_size,args.seq_length,args.rnn_input_size_1])

		# RNN setting
		outputs_1, last_state_1 = tf.nn.dynamic_rnn(cell1,input_rnn1,initial_state = initial_state1,dtype = tf.float32,scope = 'LSTM1')
		outputs_2, last_state_2 = tf.nn.dynamic_rnn(cell1,input_rnn2,initial_state = initial_state2,dtype = tf.float32,scope = 'LSTM2')

		# concatenate two outputs
		outputs_1 = tf.reshape(outputs_1,[-1,args.rnn_size])
		outputs_2 = tf.reshape(outputs_2,[-1,args.rnn_size])

		concat_output = tf.concat([outputs_1,outputs_2],1)

		# softmax of outputs
		outputs_mor = tf.matmul(concat_output,softmax_w3)+softmax_b3
		outputs_word = tf.matmul(concat_output,softmax_w4)+softmax_b4

		#=================#
		## loss function ##
		#=================#

		self.probs_word = tf.nn.softmax(outputs_word)
		self.probs_mor = tf.nn.softmax(outputs_mor)
		self.final_state1 = last_state_1
		self.final_state2 = last_state_2


		probs_word_argmax = tf.argmax(self.probs_word,1)
		probs_word_type = tf.to_int32(tf.gather(chars_type,probs_word_argmax))
		probs_word_type = tf.reshape(probs_word_type,[args.batch_size,args.seq_length])

		# make weights matrix for sequence_loss(word)
		# if morpheme of result word is not equal to result morpheme, we give massive panelty. 	
		outputs_word = tf.reshape(outputs_word,[args.batch_size,args.seq_length,args.vocab_size])
		outputs_mor = tf.reshape(outputs_mor,[args.batch_size,args.seq_length,args.vocab_mor_size])
		weights_word = tf.to_float(tf.not_equal(probs_word_type,target_mor))
		weights_word = tf.exp(weights_word)
		weights_word_multiply = tf.constant(args.morpheme_weight,dtype = tf.float32,shape = [args.batch_size,args.seq_length])
		weights_word = tf.multiply(weights_word,weights_word_multiply)
		weights_mor = tf.ones([args.batch_size,args.seq_length])

		## we need to modify the loss function.
		word_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_word, targets = target_word, weights = weights_word)
		mor_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_mor, targets = target_mor, weights = weights_mor)
		self.mean_loss = (tf.reduce_mean(word_loss) + args.morpheme_weight * tf.reduce_mean(mor_loss))/(args.morpheme_weight + 1)

		# set train rate 
		self.lr = tf.Variable(0.0, trainable = False)
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.mean_loss)

	# ==================== #
	# === writing part === #
	# ==================== #
	def sample(self,sess,chars,vocab,chars_mor,vocab_mor,num=200,prime='나는 ', sampling_type = 1):
		state1 = sess.run(self.cell1.zero_state(1,tf.float32))
		state2 = sess.run(self.cell2.zero_state(1,tf.float32))

		kkma=Kkma()
		word_info = kkma.pos(prime)

		i = 0 # word_info index
		j = 0 # prime index

		temp_tensor = []
		while 1:
			if i >= len(word_info):
				break
			if prime[j] == ' ':
				temp_tensor.append((vocab[' '],vocab_mor['P']))	
				j=j+1
				continue
			if prime[j] =='\r':
				temp_tensor.append((vocab['\r\n'],vocab_mor['L']))
				j=j+2
				continue

			temp_word = word_info[i][0]
			temp_word_mor = word_info[i][1][0]
			temp_word_length = len(temp_word)
			if temp_word == prime[j:j+temp_word_length]:
				temp_tensor.append((vocab[temp_word],vocab_mor[temp_word_mor]))
				j = j+temp_word_length
				i = i+1
			else:
				## 동사변형 bypass method

				s=0
				flag = 0
				while 1:
					s = s+1
					total_search_range = 0
					for k in range(s+1):
						if i+k == len(word_info):
							flag = 2
							break
						total_search_range = total_search_range + len(word_info[i+k][0])
					total_search_range = total_search_range + s # 여유분

					if flag == 2:
						break

					for k in range(total_search_range+1):
						bypass_word_length = len(word_info[i+s][0])
						if word_info[i+s][0] == prime[j+k:j+k+bypass_word_length]:
							for s2 in range(s):
								temp_tensor.append((vocab[word_info[i+s2][0]],vocab_mor[word_info[i+s2][1][0]]))
							if prime[j+k-1:j+k] == ' ':
								temp_tensor.append((vocab[' '],vocab_mor['P']))
							flag = 1
							break

					if flag == 1:
						j = j+k
						i = i+s
						break	
			
		mor_list = []
		char_list = []	
		for i in range(len(temp_tensor)-1):
			x10 = np.zeros((1,1))
			x10[0,0] = temp_tensor[i][0]
			x11 = np.zeros((1,1))
			x11[0,0] = temp_tensor[i][1]
			mor_list.append(chars_mor[temp_tensor[i][1]])
			char_list.append(chars[temp_tensor[i][0]])


			feed = {self.input_data_word : x10, self.input_data_mor : x11, self.initial_state1 : state1, self.initial_state2 : state2}
			[state1, state2] = sess.run([self.final_state1,self.final_state2],feed)

		# 수정(?)	
		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t,np.random.rand(1)*s)))	


		ret = prime

		char = temp_tensor[-1][0]
		mor = temp_tensor[-1][1]
		count = 0
		for _ in range(num):
			x10 = np.zeros((1,1))
			x11 = np.zeros((1,1))
			x10[0,0] = char
			x11[0,0] = mor

			feed = {self.input_data_word : x10, self.input_data_mor : x11, self.initial_state1 : state1, self.initial_state2 : state2}
			[probs_word,probs_mor,state1,state2] = sess.run([self.probs_word,self.probs_mor,self.final_state1,self.final_state2],feed)

			p_w = probs_word[0]
			p_m = probs_mor[0]

			sample_w = weighted_pick(p_w)
			sample_m = weighted_pick(p_m)
			
			count += 1
			if count % 100 == 0:
				print('%d/%d' % (count,num))

			pred_w = chars[sample_w]
			pred_m = chars_mor[sample_m]
			ret += pred_w
			char_list.append(pred_w)
			mor_list.append(pred_m)
			char = vocab[pred_w]
			mor = vocab_mor[pred_m]
		return ret,char_list,mor_list	