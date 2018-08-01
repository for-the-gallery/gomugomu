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
		
		with tf.variable_scope("mor_LSTM"):
			self.cell1 = cell1 = tf.contrib.rnn.MultiRNNCell(cells_mor,state_is_tuple = True)
		with tf.variable_scope("word_LSTM"):	
			self.cell2 = cell2 = tf.contrib.rnn.MultiRNNCell(cells_word,state_is_tuple = True)
		#self.cell_for_word = cell_for_word = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple = True)
		
		'''
		self.cell1 = cell1 = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size,state_is_tuple = True)
		self.cell2 = cell2 = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size,state_is_tuple = True)
		'''

		# data input		
		input_data_word = self.input_data_word = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		input_data_mor = self.input_data_mor = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_word = self.target_word = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		target_mor = self.target_mor = tf.placeholder(tf.int32,[args.batch_size,args.seq_length])
		chars_type = self.chars_type = tf.placeholder(tf.int32,[args.vocab_size])

		self.initial_state_word = initial_state_word = cell1.zero_state(args.batch_size,tf.float32)
		self.initial_state_mor = initial_state_mor = cell2.zero_state(args.batch_size,tf.float32)

		# one-hot
		input_onehot_word = tf.one_hot(input_data_word,args.vocab_size)
		input_onehot_mor = tf.one_hot(input_data_mor,args.vocab_mor_size)

		#====================#
		## softmax variable ##
		#====================#
		# for first RNN result(morpheme softmax layer)
		softmax_w1 = tf.get_variable("softmax_for_mor_w",[args.rnn_size,args.vocab_mor_size])
		softmax_b1 = tf.get_variable("softmax_for_mor_b",[args.vocab_mor_size])

		# for second RNN result(word softmax layer)
		softmax_w2 = tf.get_variable("softmax_for_word_w",[args.rnn_size,args.vocab_size])
		softmax_b2 = tf.get_variable("softmax_for_word_b",[args.vocab_size])

		# for second RNN input
		softmax_w3 = tf.get_variable("softmax_for_input_w",[args.vocab_mor_size+args.vocab_size,args.vocab_size])
		softmax_b3 = tf.get_variable("softmax_for_input_b",[args.vocab_size])


		##################
		# size test, debuging...
		#print(tf.size(input_onehot_mor))
		##################

		#==============#
		## RNN design ##
		#==============#
		# first RNN, morpheme to morpheme
		outputs_mor,last_state_mor = tf.nn.dynamic_rnn(cell1,input_onehot_mor,initial_state = initial_state_mor,dtype = tf.float32,scope = 'mor_LSTM')
		outputs_mor = tf.reshape(outputs_mor,[-1,args.rnn_size])
		outputs_mor_final = tf.matmul(outputs_mor,softmax_w1)+softmax_b1

		# second RNN, word to word
		outputs_word,last_state_word = tf.nn.dynamic_rnn(cell2,input_onehot_word,initial_state = initial_state_word,dtype = tf.float32,scope = 'word_LSTM')
		outputs_word = tf.reshape(outputs_word,[-1,args.rnn_size])
		outputs_word_final = tf.matmul(outputs_word,softmax_w2)+softmax_b2

		# word_result * mor_result
		outputs_concat = tf.concat([outputs_word_final,outputs_mor_final],1)
		outputs_final = tf.matmul(outputs_concat,softmax_w3) + softmax_b3


		#=================#
		## loss function ##
		#=================#

		self.probs_word = tf.nn.softmax(outputs_final)
		self.probs_mor = tf.nn.softmax(outputs_mor_final)
		self.final_state_mor = last_state_mor
		self.final_state_word = last_state_word


		probs_word_argmax = tf.argmax(self.probs_word,1)
		probs_word_type = tf.to_int32(tf.gather(chars_type,probs_word_argmax))
		probs_word_type = tf.reshape(probs_word_type,[args.batch_size,args.seq_length])


		outputs_final = tf.reshape(outputs_final,[args.batch_size,args.seq_length,args.vocab_size])
		outputs_mor_final = tf.reshape(outputs_mor_final,[args.batch_size,args.seq_length,args.vocab_mor_size])
		weights_word = tf.to_float(tf.not_equal(probs_word_type,target_mor))
		weights_word = tf.exp(weights_word)
		weights_word_multiply = tf.constant(args.morpheme_weight,dtype = tf.float32,shape = [args.batch_size,args.seq_length])
		weights_word = tf.multiply(weights_word,weights_word_multiply)
		weights_mor = tf.ones([args.batch_size,args.seq_length])

		## modify the loss function.
		# word_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_final, targets = target_word, weights = weights)
		# mor_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_mor_final, targets = target_mor, weights = weights)
		# self.mean_loss = (tf.reduce_mean(word_loss) + args.morpheme_weight * tf.reduce_mean(mor_loss))/(args.morpheme_weight + 1)
		word_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_final, targets = target_word, weights = weights_word)
		mor_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs_mor_final, targets = target_mor, weights = weights_mor)
		self.mean_loss = (tf.reduce_mean(word_loss) + args.morpheme_weight * tf.reduce_mean(mor_loss))/(args.morpheme_weight + 1)

		# train
		self.lr = tf.Variable(0.0, trainable = False)
		self.train_op = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.mean_loss)

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
		for i in range(len(temp_tensor)-1):
			x10 = np.zeros((1,1))
			x10[0,0] = temp_tensor[i][0]
			x11 = np.zeros((1,1))
			x11[0,0] = temp_tensor[i][1]
			mor_list += chars_mor[temp_tensor[i][1]]


			feed = {self.input_data_word : x10, self.input_data_mor : x11, self.initial_state_word : state1, self.initial_state_mor : state2}
			[state1, state2] = sess.run([self.final_state_word,self.final_state_mor],feed)

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

			feed = {self.input_data_word : x10, self.input_data_mor : x11, self.initial_state_word : state1, self.initial_state_mor : state2}
			[probs_word,probs_mor,state1,state2] = sess.run([self.probs_word,self.probs_mor,self.final_state_word,self.final_state_mor],feed)

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
			mor_list += pred_m
			char = vocab[pred_w]
			mor = vocab_mor[pred_m]
		return ret,mor_list	



