import tensorflow as tf 
import numpy as np 

class Model():
	def __init__(self,args,training = True):
		self.args = args
		if not training:
			args.batch_size = 1
			args.seq_length = 1

		# data input
		input_data = self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
		targets = self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

		input_onehot = tf.one_hot(input_data, args.vocab_size)	

		# cell design -> LSTM multi layer
		cells = []
		for _ in range(args.num_layers):
			cell = tf.contrib.rnn.BasicLSTMCell(num_units = args.rnn_size, state_is_tuple = True)
			cells.append(cell)
		self.cell = cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple = True)

		self.initial_state = initial_state = cell.zero_state(args.batch_size,tf.float32)
		outputs,last_state = tf.nn.dynamic_rnn(cell,input_onehot,initial_state = initial_state,
			dtype = tf.float32)

		# softmax_layer
		output_for_softmax = tf.reshape(outputs,[-1,args.rnn_size])

		softmax_w = tf.get_variable("softmax_w",[args.rnn_size, args.vocab_size])
		softmax_b = tf.get_variable("softmax_b",[args.vocab_size])
		outputs = tf.matmul(output_for_softmax,softmax_w) + softmax_b

		self.probs = tf.nn.softmax(outputs)
		self.final_state = last_state

		outputs = tf.reshape(outputs,[args.batch_size,args.seq_length,args.vocab_size])
		weights = tf.ones([args.batch_size,args.seq_length])

		# loss
		sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets = targets, weights = weights)
		self.mean_loss = tf.reduce_mean(sequence_loss)

		self.lr = tf.Variable(0.0, trainable = False)
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.mean_loss)


	def sample(self,sess,chars,vocab,num=200,prime='The ', sampling_type = 1):
		state = sess.run(self.cell.zero_state(1, tf.float32))
		for char in prime[:-1]:
			x = np.zeros((1,1))
			x[0, 0] = vocab[char]
			feed = {self.input_data: x, self.initial_state: state}
			[state] = sess.run([self.final_state], feed)

		def weighted_pick(weights):
			t = np.cumsum(weights)
			s = np.sum(weights)
			return(int(np.searchsorted(t, np.random.rand(1)*s)))	

		for _ in range(num):
			x = np.zeros((1,1))
			x[0,0] = vocab[char]
			feed = {self.input_data : x, self.initial_state: state}
			[probs, state] = sess.run([self.probs, self.final_state], feed)
			p = probs[0]

			sample = weighted_pick(p)

			pred = chars[sample]
			ret += pred
			char = pred
		return ret	



		



