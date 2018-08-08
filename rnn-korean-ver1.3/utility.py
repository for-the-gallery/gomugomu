import codecs
import os
import time
import numpy as np 
from six.moves import cPickle
from konlpy.tag import Kkma

class Textloader():
	def __init__(self,data_dir,batch_size,seq_length,encoding = 'cp949'):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding

		input_file = os.path.join(data_dir,'input.txt')
		vocab_file = os.path.join(data_dir,'vocab.pkl')
		vocab_mor_file = os.path.join(data_dir,'vocab_mor.pkl')
		chars_type_file = os.path.join(data_dir,'chars_type.pkl')
		tensor_file = os.path.join(data_dir,'tensor_data.npy')

		if not (os.path.exists(vocab_file) and os.path.exists(vocab_mor_file) and os.path.exists(tensor_file)):	
			print("reading text file")
			self.preprocess(input_file,vocab_file,vocab_mor_file,chars_type_file,tensor_file)
		else:
			print("loading tensor file")
			self.load_preprocess(vocab_file,vocab_mor_file,chars_type_file,tensor_file)

		self.create_batches()
		self.reset_batch_pointer()


	def preprocess(self,input_file,vocab_file,vocab_mor_file,chars_type_file,tensor_file):
		with codecs.open(input_file,"r",encoding = self.encoding) as f:
			data = f.read()

		# process konlpy		
		print("classify the morpheme of each word...")	
		start_time = time.time()
		kkma = Kkma()
		len_data = len(data)
		word_info = kkma.pos(data)
		print("--- %s seconds ---" % (time.time()-start_time))

		## make vocab file and vocab_mor file
		print("produce the vocab.pkl")
		
		i=0
		chars = []
		chars_mor = []
		temp_chars_type = []
		space_symbol = ' ','P'
		linebreak_symbol = '\r\n','LB'
		for _ in range(len(word_info)):
			if i ==0:
				chars.append(word_info[i][0])
				temp_chars_type.append(word_info[i][1][0])
				chars_mor.append(word_info[i][1][0])
			else:
				if not word_info[i][0] in chars:
					chars.append(word_info[i][0])
					temp_chars_type.append(word_info[i][1][0])
				if not word_info[i][1][0] in chars_mor:
					chars_mor.append(word_info[i][1][0])
			i = i+1
			
		chars.append(space_symbol[0])
		chars.append(linebreak_symbol[0])
		temp_chars_type.append(space_symbol[1][0])
		temp_chars_type.append(linebreak_symbol[1][0])
		chars_mor.append(space_symbol[1][0])
		chars_mor.append(linebreak_symbol[1][0])


		with open(vocab_file, 'wb') as f:
			cPickle.dump(chars,f)
		with open(vocab_mor_file,'wb') as f:
			cPickle.dump(chars_mor,f)	

		## make tensor file		
		print("produce tensor.npy")
		
		vocab = dict(zip(chars,range(len(chars))))
		vocab_mor = dict(zip(chars_mor,range(len(chars_mor))))

		chars_type = []
		for i in range(len(temp_chars_type)):
			chars_type.append(vocab_mor[temp_chars_type[i]])

		with open(chars_type_file,'wb') as f:
			cPickle.dump(chars_type,f)	


		i = 0 # word_info index
		j = 0 # data index
		temp_tensor = []

		while 1:
			if i >= len(word_info):
				break

			if data[j] == ' ':
				temp_tensor.append((vocab[' '],vocab_mor['P']))
				j=j+1
				continue
			if data[j] =='\r':
				temp_tensor.append((vocab['\r\n'],vocab_mor['L']))
				j=j+2
				continue

			temp_word = word_info[i][0]
			temp_word_mor = word_info[i][1][0]
			temp_word_length = len(temp_word)
			if temp_word == data[j:j+temp_word_length]:
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
						if word_info[i+s][0] == data[j+k:j+k+bypass_word_length]:
							for s2 in range(s):
								temp_tensor.append((vocab[word_info[i+s2][0]],vocab_mor[word_info[i+s2][1][0]]))
							if data[j+k-1:j+k] == ' ':
								temp_tensor.append((vocab[' '],vocab_mor['P']))
							flag = 1
							break

					if flag == 1:
						j = j+k
						i = i+s
						break		
		tensor = np.array(temp_tensor)
		np.save(tensor_file, tensor)

		# save at self
		self.chars = chars
		self.vocab_size = len(self.chars)
		self.vocab = vocab
		self.chars_mor = chars_mor
		self.vocab_mor_size = len(self.chars_mor)
		self.vocab_mor = vocab_mor
		self.chars_type = chars_type
		self.tensor = tensor	

	def load_preprocess(self,vocab_file,vocab_mor_file,chars_type_file,tensor_file):
		with open(vocab_file,'rb') as f:
			self.chars = cPickle.load(f)
		with open(vocab_mor_file,'rb') as f:
			self.chars_mor = cPickle.load(f)
		with open(chars_type_file,'rb') as f:
			self.chars_type = cPickle.load(f)

		self.vocab_size = len(self.chars)	
		self.vocab = dict(zip(self.chars,range(len(self.chars))))
		self.vocab_mor_size = len(self.chars_mor)
		self.vocab_mor = dict(zip(self.chars_mor,range(len(self.chars_mor))))
		self.tensor = np.load(tensor_file)

	def create_batches(self):
		self.num_batches = int(self.tensor.size/2/(self.batch_size*self.seq_length))

		if self.num_batches == 0:
			assert False, "Not Enough Data."

		# reshape the original data 	
		self.tensor = self.tensor[:self.num_batches*self.batch_size*self.seq_length]
		xdata = self.tensor

		# data devide to word and morpheme
		xdata1 = [] #word
		xdata2 = []	#morpheme
		for i in range(len(xdata)):
			xdata1.append(xdata[i][0])
			xdata2.append(xdata[i][1])
		xdata1 = np.array(xdata1)
		xdata2 = np.array(xdata2)	
		ydata1 = np.copy(xdata1)
		ydata2 = np.copy(xdata2)

		# ydata is the xdata with one position shift.
		ydata1[:-1] = xdata1[1:]
		ydata1[-1] = xdata1[0]
		ydata2[:-1] = xdata2[1:]
		ydata2[-1] = xdata2[0]

		self.x_batches_1 = np.split(xdata1.reshape(self.batch_size,-1),self.num_batches,1)
		self.x_batches_2 = np.split(xdata2.reshape(self.batch_size,-1),self.num_batches,1)
		self.y_batches_1 = np.split(ydata1.reshape(self.batch_size,-1),self.num_batches,1)
		self.y_batches_2 = np.split(ydata2.reshape(self.batch_size,-1),self.num_batches,1)

	def next_batch(self):
		x1, x2, y1, y2 = self.x_batches_1[self.pointer],self.x_batches_2[self.pointer],self.y_batches_1[self.pointer],self.y_batches_2[self.pointer]
		self.pointer += 1
		return x1,x2,y1,y2

	def reset_batch_pointer(self):
		self.pointer = 0	



					

					




