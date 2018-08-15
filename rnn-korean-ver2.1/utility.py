# -*- coding: utf-8 -*-
import codecs
import os
import time
import numpy as np 
import unicode_converter as uc 

from six.moves import cPickle


class Textloader():
	def __init__(self,data_dir,batch_size,seq_length,encoding = 'cp949'):
		self.data_dir = data_dir
		self.batch_size = batch_size
		self.seq_length = seq_length
		self.encoding = encoding

		input_file = os.path.join(data_dir,'input.txt')
		vocab_file1 = os.path.join(data_dir,'vocab1.pkl')
		vocab_file2 = os.path.join(data_dir,'vocab2.pkl')
		vocab_file3 = os.path.join(data_dir,'vocab3.pkl')
		tensor_file = os.path.join(data_dir,'tensor_data.npy')

		if not (os.path.exists(vocab_file1) and os.path.exists(vocab_file2) and os.path.exists(vocab_file3) and os.path.exists(tensor_file)):	
			print("reading text file")
			self.preprocess(input_file,vocab_file1,vocab_file2,vocab_file3,tensor_file)
		else:
			print("loading tensor file")
			self.load_preprocess(vocab_file1,vocab_file2,vocab_file3,tensor_file)

		self.create_batches()
		self.reset_batch_pointer()


	def preprocess(self,input_file,vocab_file1,vocab_file2,vocab_file3,tensor_file):
		with codecs.open(input_file,"r",encoding = self.encoding) as f:
			data = f.read()

		self.first_arr = first_arr = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ',
			'ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
		self.second_arr = second_arr = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ',
			'ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
		self.third_arr = third_arr = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ',
			'ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ',
			'ㅌ','ㅍ','ㅎ']

		with codecs.open(vocab_file1,"wb") as f:
			cPickle.dump(self.first_arr,f)	
		with codecs.open(vocab_file2,"wb") as f:
			cPickle.dump(self.second_arr,f)	
		with codecs.open(vocab_file3,"wb") as f:
			cPickle.dump(self.third_arr,f)	

		vocab_1 = dict(zip(first_arr,range(len(first_arr))))
		vocab_2 = dict(zip(second_arr,range(len(second_arr))))	
		vocab_3 = dict(zip(third_arr,range(len(third_arr))))

		## make tensor file		
		print("produce tensor.npy")

		temp_tensor = []
		for i in range(len(data)):
			if data[i] == ' ':
				continue
			else:
				char1,char2,char3 = uc.chr_diss(data[i])
				if isinstance(char1,int):
					continue	
				temp_tensor.append((vocab_1[char1],vocab_2[char2],vocab_3[char3]))	

		tensor = np.array(temp_tensor)
		np.save(tensor_file, tensor)

		# save at self
		self.vocab1_size = len(self.first_arr)
		self.vocab2_size = len(self.second_arr)
		self.vocab3_size = len(self.third_arr)

		self.vocab_1 = vocab_1
		self.vocab_2 = vocab_2
		self.vocab_3 = vocab_3

		self.tensor = tensor	

	def load_preprocess(self,vocab_file1,vocab_file2,vocab_file3,tensor_file):
		with open(vocab_file1,'rb') as f:
			self.first_arr = cPickle.load(f)
		with open(vocab_file2,'rb') as f:
			self.second_arr = cPickle.load(f)
		with open(vocab_file3,'rb') as f:
			self.third_arr = cPickle.load(f)

		vocab_1 = dict(zip(self.first_arr,range(len(self.first_arr))))
		vocab_2 = dict(zip(self.second_arr,range(len(self.second_arr))))	
		vocab_3 = dict(zip(self.third_arr,range(len(self.third_arr))))	

		self.vocab1_size = len(self.first_arr)
		self.vocab2_size = len(self.second_arr)
		self.vocab3_size = len(self.third_arr)

		self.vocab_1 = vocab_1
		self.vocab_2 = vocab_2
		self.vocab_3 = vocab_3

		self.tensor = np.load(tensor_file)

	def create_batches(self):
		self.num_batches = int(self.tensor.size/3/(self.batch_size*self.seq_length))

		if self.num_batches == 0:
			assert False, "Not Enough Data."

		# reshape the original data 	
		self.tensor = self.tensor[:self.num_batches*self.batch_size*self.seq_length]
		xdata = self.tensor

		# data devide to initial/medial/final sound
		xdata1 = []
		xdata2 = []
		xdata3 = []

		for i in range(len(xdata)):
			xdata1.append(xdata[i][0])
			xdata2.append(xdata[i][1])
			xdata3.append(xdata[i][2])	

		xdata1 = np.array(xdata1)
		xdata2 = np.array(xdata2)
		xdata3 = np.array(xdata3)	
		ydata1 = np.copy(xdata1)
		ydata2 = np.copy(xdata2)
		ydata3 = np.copy(xdata3)

		# ydata is the xdata with one position shift.
		ydata1[:-1] = xdata1[1:]
		ydata1[-1] = xdata1[0]
		ydata2[:-1] = xdata2[1:]
		ydata2[-1] = xdata2[0]
		ydata3[:-1] = xdata3[1:]
		ydata3[-1] = xdata3[0]

		self.x_batches_1 = np.split(xdata1.reshape(self.batch_size,-1),self.num_batches,1)
		self.x_batches_2 = np.split(xdata2.reshape(self.batch_size,-1),self.num_batches,1)
		self.x_batches_3 = np.split(xdata3.reshape(self.batch_size,-1),self.num_batches,1)
		self.y_batches_1 = np.split(ydata1.reshape(self.batch_size,-1),self.num_batches,1)
		self.y_batches_2 = np.split(ydata2.reshape(self.batch_size,-1),self.num_batches,1)
		self.y_batches_3 = np.split(ydata3.reshape(self.batch_size,-1),self.num_batches,1)

	def next_batch(self):
		x1, x2, x3, y1, y2, y3 = self.x_batches_1[self.pointer],self.x_batches_2[self.pointer],self.x_batches_3[self.pointer],self.y_batches_1[self.pointer],self.y_batches_2[self.pointer],self.y_batches_3[self.pointer]
		self.pointer += 1
		return x1,x2,x3,y1,y2,y3

	def reset_batch_pointer(self):
		self.pointer = 0	



					

					




