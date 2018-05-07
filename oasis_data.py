import os
import torch
import numpy as np
from random import randint

import sys
sys.path.insert(0, "~/Documents/spring2018/543/project/code/")
from util import *

class OasisDataset(torch.utils.data.Dataset):
	def __init__(self, type, num_channels=5):
		super(OasisDataset, self).__init__()

		self.data_path = './pickled/unlabelled/'
		self.label_path = './pickled/labelled/'

		self.num_channels = num_channels

		self.data_files = sorted(os.listdir(self.data_path))
		self.label_files = sorted(os.listdir(self.label_path))

		self.type = type
		if type == 'train':
			self.range = (1,14)
		elif type == 'val':
			self.range = (15, 15)
		elif type == 'test':
			self.range = (16,20)

		self.data_cache_ = {}
		self.label_cache_ = {}

	def __len__(self):
		return(self.range[1] - self.range[0] + 1)

	def __getitem__(self, item):

		subject = randint(self.range[0], self.range[1])
		if subject in self.data_cache_.keys():
			data = self.data_cache_[subject]
			label = self.label_cache_[subject]
		else:
			data = np.load(self.data_path + self.data_files[subject-1]).astype(np.float32)
			label = np.load(self.label_path + self.label_files[subject-1]).astype(np.int64)

			self.data_cache_[subject] = data
			self.label_cache_[subject] = label

		if self.type == 'train':

			buff = (self.num_channels - 1) // 2
			middle_slice = randint(0+buff, 255-buff)
			data = data[middle_slice-buff:middle_slice+buff+1, :, :]
			label = label[middle_slice, :, :]

		return(torch.from_numpy(data), torch.from_numpy(label))