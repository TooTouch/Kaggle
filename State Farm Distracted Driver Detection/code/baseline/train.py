from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.utils import to_categorical
from keras import applications as app
from keras import callbacks as cb
from keras import optimizers as op
from keras import layers
from keras import models

import os
import numpy as np
import cv2
import h5py
import argparse
import time
from tqdm import tqdm
from pprint import pprint

from collections import OrderedDict
import json

class State_Train:
	def __init__(self, json_file):
		# directory
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'../..'))
		param_dir = os.path.join(self.root_dir + '/training_params/', json_file)
		train_h5py = os.path.join(self.root_dir, 'dataset/h5py/train')
		self.model_dir = os.path.join(self.root_dir, 'model')
		self.image_dir = os.path.join(self.root_dir, 'dataset/train')
		self.class_id = os.listdir(self.image_dir)
		self.log_dir = os.path.join(self.root_dir, 'log')
		# open parameters 
		f = open(param_dir)
		params = json.load(f)
		# image_data
		train_h5py += str(tuple(params['IMAGE_SIZE']))
		self.hf = h5py.File(train_h5py,'r')
		print('='*100)
		print('Image directory: ',train_h5py)
		# parameter
		print('='*100)
		print('Parameters')
		pprint(params)
		print('='*100)
		self.id = params['ID']
		self.name = params['NAME']
		self.save_name = self.name + str(self.id)
		self.class_num =  params['CLASS_NUM']
		self.cv_rate = params['CV_RATE']
		self.cv_seed = params['CV_SEED']
		self.epochs = params['EPOCHS']
		self.batch_size = params['BATCH_SIZE']
		self.optimizer = params['OPTIMIZER']
		self.self_training = params['SELFTRAINING']
		self.filename = params['LABEL_NAME']

		# model 
		self.model = None

	def run(self):
		# images
		if not(self.self_training):
			print('Load image')
			train_imgs = np.array(self.hf['train_imgs'])
			train_labels = np.array(self.hf['train_labels'])
			val_imgs = np.array(self.hf['val_imgs'])
			val_labels = np.array(self.hf['val_labels'])

			train = self.data_generator(images=train_imgs, labels=train_labels, batch_size=self.batch_size, seed=self.cv_seed)
			val = self.data_generator(images=val_imgs, labels=val_labels, batch_size=self.batch_size, seed=self.cv_seed)

			print('='*100)
			print('Train dataset shape: ',train_imgs.shape)
			print('Validation dataset shape: ',val_imgs.shape)

			# size
			size = (train_imgs.shape[0], val_imgs.shape[0])
			# delete
			del(train_imgs); del(train_labels); del(val_imgs); del(val_labels);
			# close
			self.hf.close()
			
			# model
			self.model_()
			# train
			history, train_time = self.training(train=train, val=val, size=size)
			# report
			self.report_json(history=history, time=train_time)

		# self training
		elif self.self_training:
			print('Self training')			
			h5py_dir = self.root_dir + '/dataset/h5py/'
			
			submit_dir = h5py_dir + self.filename
			submit_h5py = h5py.File(submit_dir,'r')
			test_dir = h5py_dir + 'test(224, 224, 3)'
			test_h5py = h5py.File(test_dir,'r')

			val_imgs = np.array(self.hf['val_imgs'])
			val_labels = np.array(self.hf['val_labels'])
			
			train_labels = np.array(submit_h5py['test_labels'])

			# model
			self.model_()

			for i in range(10):
				print('-'*100)
				print('{} train'.format(i))
				train_imgs_c = np.array(test_h5py['test'+str(i)])
				train_labels_c = train_labels[:train_imgs_c.shape[0]]
				train_labels = train_labels[train_imgs_c.shape[0]:]
				
				train = self.data_generator(images=train_imgs_c, labels=train_labels_c, batch_size=self.batch_size, seed=self.cv_seed)
				val = self.data_generator(images=val_imgs, labels=val_labels, batch_size=self.batch_size, seed=self.cv_seed)

				print('='*100)
				print('Train dataset shape: ',train_imgs_c.shape)
				print('Validation dataset shape: ',val_imgs.shape)

				# size
				size = (train_imgs_c.shape[0], val_imgs.shape[0])
				# delete
				del(train_imgs_c); del(train_labels_c);
				# close
				self.hf.close()
				# model
				self.model_(number=i)
				# train
				history, train_time = self.training(train=train, val=val, size=size)
			

			test_h5py.close()
			submit_h5py.close()
			print('Complete')

			
		

	def data_generator(self, images, labels, batch_size, seed=None, shuffle=True):
		datagen = ImageDataGenerator(
					rescale = 1./255,
					rotation_range = 10
					)
		iterator = datagen.flow(
			x = images,
			y = labels,
			batch_size = batch_size,
			seed=seed,
			shuffle=shuffle
		)
		for batch_x, batch_y in iterator:
			yield batch_x, batch_y


	def model_(self, number=0):
		print('='*100)
		print('Load Model')
		print('Model name: ', self.save_name)
		print('-'*100)
		if self.self_training:
			self.model = models.load_model(self.model_dir + '/' + self.save_name + '.h5')
			adam = op.Adam(lr=0.00001)
			self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
		else:
			a = app.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='avg', classes=10)
			x = a.output
			output = layers.Dense(10, activation='softmax')(x)
			self.model = models.Model(a.input, output, name='Pretrain_inception_resnet_v2')

			adam = op.Adam(lr=0.0001)
			self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['acc'])
		print('Complete')
		print('='*100)
			

	def training(self, train, val, size, weights=None):
		print('='*100)
		print('Start training')
		print('-'*100)
		if self.self_training:
			if 'selftraining' not in self.save_name:
				self.save_name += '_selftraining'
				
		ckp = cb.ModelCheckpoint(self.model_dir + '/' + self.save_name + '.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
		es = cb.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
		rlp = cb.ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.00001)
		start = time.time()
		history = self.model.fit_generator(generator=train,
							steps_per_epoch=size[0]//self.batch_size * 3,
							epochs=self.epochs,
							validation_data=val,
							validation_steps=size[1]//self.batch_size,
							class_weight=weights,
							callbacks=[es,ckp,rlp]
						   )
		e = int(time.time() - start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		train_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
		print('Train time: ',train_time)
		print('='*100)
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

		print('='*100)
		print('Save log to json file')
		with open(self.log_dir + '/' + self.save_name + '_info.json','w',encoding='utf-8') as make_file:
			json.dump(log, make_file, ensure_ascii=False, indent='\t')
		print('Complete')
		print('='*100)


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--params', type=str, help='Json file name containig paramters')
	args = parser.parse_args()

	ST = State_Train(json_file=args.params)
	ST.run()