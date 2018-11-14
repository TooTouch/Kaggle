from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras import applications as app
from keras import callbacks as cb

import os
import numpy as np
import h5py
import json
import argparse
import time
from pprint import pprint

class State_Train:
	def __init__(self, json_file):
		# directory
		root_dir = os.path.abspath(os.path.join(os.getcwd(),'../..'))
		param_dir = os.path.join(root_dir + '/training_params/', json_file)
		train_h5py = os.path.join(root_dir, 'dataset/h5py/train')
		self.model_dir = os.path.join(root_dir, 'model')
		self.image_dir = os.path.join(root_dir, 'dataset/train')
		self.log_dir = os.path.join(root_dir, 'log')
		# image_data
		self.hf = h5py.File(train_h5py,'r')
		# parameter
		f = open(param_dir)
		params = json.load(f)
		print('='*50)
		print('Parameters')
		pprint(params)
		print('='*50)
		self.id = params['ID']
		self.name = params['NAME']
		self.save_name = str(self.id) + self.name
		self.class_num =  params['CLASS_NUM']
		self.cv_rate = params['CV_RATE']
		self.cv_seed = params['CV_SEED']
		self.epochs = params['EPOCHS']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']

		# model 
		self.model = None

	def run(self):
		# images
		c_imgs, c_labels = self.image_load()
		train, train_size, val, val_size = self.split_data(images = c_imgs, labels = c_labels)
		# model
		self.model_()
		# train
		history, train_time = self.training(train=train, val=val, size=(train_size,val_size))
		# report
		self.report_json(history=history, time=train_time)


	def image_load(self):
		c_imgs = list()
		c_labels = list()
		c_list = os.listdir(self.image_dir)
		print('='*50)
		print('Load images')
		for i in range(len(c_list)):
			# load images
			c_npy = self.hf['train'+str(i)]
			c_imgs.append(c_npy)
			print('-'*50)
			print('Complete to load {} images'.format(c_list[i]))
			print('shape: ',c_npy.shape)
			# label to categorical
			c_len = c_npy.shape[0]
			label = [i] * c_len
			c_label = to_categorical(label, self.class_num)
			c_labels.append(c_label)
			print('Complete to convert label to categorical')
			print('label: ',c_label[0])
		print('='*50)

		return c_imgs, c_labels


	def split_data(self, images, labels):
		train_imgs = list()
		train_labels = list()
		val_imgs = list()
		val_labels = list()
		print('='*50)
		print('Split data by class')
		print('-'*50)
		for i in range(self.class_num):
			c_len = images[i].shape[0]
			val_len = int(c_len * self.cv_rate)
			val_imgs.append(images[i][:val_len])
			val_labels.append(labels[i][:val_len])
			train_imgs.append(images[i][val_len:])
			train_labels.append(labels[i][val_len:])
			
			print('Complete to split c{}'.format(i))
			
		train_imgs = np.vstack(train_imgs)
		train_labels = np.vstack(train_labels)
		val_imgs = np.vstack(val_imgs)
		val_labels = np.vstack(val_labels)

		print('-'*50)
		train_size = train_imgs.shape
		val_size = val_imgs.shape
		print('Shape train images: ',train_size)
		print('Shape validation images: ',val_size)
		print('='*50)

		train = self.data_generator(images=train_imgs,
								   labels=train_labels,
								   batch_size=self.batch_size,
								   seed=self.cv_seed,
								   shuffle=True)
		val = self.data_generator(images=val_imgs,
								labels=val_labels,
								batch_size=self.batch_size,
								seed=self.cv_seed,
								shuffle=True)
		self.hf.close()

		return train, train_size[0], val, val_size[0]


	def data_generator(self, images, labels, batch_size, seed=None, shuffle=False):
		datagen = ImageDataGenerator(rescale = 1./255)
		iterator = datagen.flow(
			x = images,
			y = labels,
			batch_size = batch_size,
			seed=seed,
			shuffle=shuffle
		)
		for batch_x, batch_y in iterator:
			yield batch_x, batch_y


	def model_(self):
		print('='*50)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*50)
		self.model = app.InceptionResNetV2(include_top=True, weights=None, input_shape=(224,224,1), classes=10)
		self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['acc'])
		print('Complete')
		print('='*50)


	def training(self, train, val, size):
		print('='*50)
		print('Start training')
		print('-'*50)
		ckp = cb.ModelCheckpoint(self.model_dir + '/' + self.save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)

		start = time.time()
		history = self.model.fit_generator(generator=train,
							steps_per_epoch=size[0]//self.batch_size, 
							epochs=self.epochs,
							validation_data=val,
							validation_steps=size[1]//self.batch_size,
							callbacks=[ckp]
						   )
		e = int(time.time() - start)
		print('-'*50)
		print('Complete')
		print('-'*50)
		train_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
		print('Train time: ',train_time)
		print('='*50)
		return history, train_time


	def report_json(self, history, time):
		log = OrderedDict()
		h = OrderedDict()
		m_compile = OrderedDict()
		log['ID'] = self.id
		log['NAME'] = self.name
		log['TIME'] = time
		log['INPUT_SHAPE'] = self.model.input_shape
		m_compile['OPTIMIZER'] = str(type(self.model.optimizer))
		m_compile['LOSS'] = self.model.loss
		m_compile['METRICS'] = self.model.metrics_names
		log['COMPILE'] = m_compile
		h = history.params
		h['LOSS'] = history.history['loss']
		h['ACCURACY'] = history.history['acc']
		h['VAL_LOSS'] = history.history['val_loss']
		h['VAL_ACCURACY'] = history.history['val_acc']
		log['HISTORY'] = h

		print('='*50)
		print('Save log to json file')
		with open(self.log_dir + self.save_name + '_info.json','w',encoding='utf-8') as make_file:
			json.dump(log, make_file, ensure_ascii=False, indent='\t')
		print('Complete')
		print('='*50)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--params', type=str, help='Json file name containig paramters')
	args = parser.parse_args()

	ST = State_Train(json_file=args.params)
	ST.run()