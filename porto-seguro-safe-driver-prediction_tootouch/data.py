'''
load data

clean data

select feature
'''

import pandas as pd 
from sklearn import preprocessing

class data:
	def __init__(self):
		self.train = pd.read_csv('../input/train.csv')[1:]
		self.test = pf.read_csv('../input/test.csv')[1:]
		self.categorize = category()
		self.change_type()
		


	def category(self):
		category = list()
		for col in self.train.columns:
			if 'cat' in col:
				category.append(col)
		return category

	def change_type(self):
		for col in self.train.columns:
			if 'bin' in col and 'cat' in col:
				self.train[col] = self.train[col].astype('object')


	def clean_data(self):
		pass

	def one_hot(self):
		pass

	def select_feature(self):
		pass

