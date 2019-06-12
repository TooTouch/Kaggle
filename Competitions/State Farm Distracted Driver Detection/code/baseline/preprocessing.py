import pandas as pd
import numpy as np
import os
import h5py
import argparse
import time

from keras.preprocessing.image import load_img, array_to_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical
from tqdm import tqdm


class Preprocessing:
	def __init__(self):
		self.root_dir = os.path.abspath(os.path.join(os.getcwd(),'../..'))
		self.d_list = pd.read_csv(self.root_dir + '/dataset/driver_imgs_list.csv')
		self.image_dir = self.root_dir + '/dataset/train/'
		self.class_id = os.listdir(self.image_dir)
		
		print('='*100)
		print('Class_id: ',list(self.class_id))
		print('Shape list: ',self.d_list.shape)

		self.data_h5py = self.root_dir + '/dataset/h5py/train(224, 224, 3).h5'
		self.hf = h5py.File(self.data_h5py,'w')
		
	

	def run(self, test, self_training, filename):
		self.clear_mis()

		train_list, val_list = self.split_()

		self.cnt_image(train_list)
		self.cnt_image(val_list)

		print('='*100)
		print('Train dataset')
		print('-'*100)		
		train_imgs = np.array([], dtype='float32').reshape(0,224,224,3)
		train_labels = np.array([], dtype='float32').reshape(0,10)
		save_start = time.time()
		for c in self.class_id:
			print(c)
			imgs, labels = self.aug_image(image_list=train_list, fold=5, class_name=c, size=(224,224))
			train_imgs = np.vstack([train_imgs,imgs])	
			train_labels = np.vstack([train_labels, labels])
			# del
			del(imgs); del(labels); 
			print('-'*100)
		self.hf.create_dataset('train_imgs', data=train_imgs, compression='lzf')
		self.hf.create_dataset('train_labels', data=train_labels, compression='lzf')
		del(train_imgs); del(train_labels);
		save_e = int(time.time() - save_start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		print('Save time: {:02d}:{:02d}:{:02d}'.format(save_e // 3600, (save_e % 3600 // 60), save_e % 60))


		print('='*100)
		print('Validation dataset')
		print('-'*100)
		val_imgs = np.array([], dtype='float32').reshape(0,224,224,3)
		val_labels = np.array([], dtype='float32').reshape(0,10)
		save_start = time.time()
		for c in self.class_id:
			print(c)
			imgs, labels = self.aug_image(image_list=val_list, fold=False, class_name=c, size=(224,224))
			val_imgs = np.vstack([val_imgs,imgs])	
			val_labels = np.vstack([val_labels, labels])
			#del
			del(imgs); del(labels)
			print('-'*100)
		self.hf.create_dataset('val_imgs', data=val_imgs, compression='lzf')
		self.hf.create_dataset('val_labels', data=val_labels, compression='lzf')
		del(val_imgs); del(val_labels);
		save_e = int(time.time() - save_start)
		print('-'*100)
		print('Complete')
		print('-'*100)
		print('Save time: {:02d}:{:02d}:{:02d}'.format(save_e // 3600, (save_e % 3600 // 60), save_e % 60))

		self.hf.close()


		if test:
			if sefl_training:
				print('='*100)
				print('Save Test images')
				save_start = time.time()
				self.save_test()
				print('-'*100)
				print('Complete')
				print('-'*100)
				print('Save time: {:02d}:{:02d}:{:02d}'.format(save_e // 3600, (save_e % 3600 // 60), save_e % 60))

		if self_training:
			self.labeling(filename=filename)


	def clear_mis(self):
		mis = self.root_dir + '/dataset/오분류/'
		mis_filenames = list()
		mis_index = list()
		for i in self.class_id:
			mis_filenames += os.listdir(os.path.join(mis,i))
		for i in mis_filenames:
			mis_index.extend(list(self.d_list[self.d_list['img']==i].index))
		self.d_list = self.d_list.drop(mis_index)
		print('='*100)
		print('Remove misclassification images')
		print('Shape list: ',self.d_list.shape)
			

	def split_(self):
		p_cnt = self.d_list['subject'].value_counts()
		p_val = p_cnt[-7:]
		print('='*100)
		print('Number of people: ',np.sum(p_cnt))
		print('Number of validation set(rate:0.2): ',int(np.sum(p_cnt) * 0.2))
		print('Validation ID: ',list(p_val.index))
		print('Validation set: ',np.sum(p_val))

		val_list = pd.DataFrame([],columns=self.d_list.columns)
		for i in p_val.index:
			val_list = pd.concat([val_list,self.d_list[self.d_list['subject']==i]], axis=0)
		train_list = pd.DataFrame([],columns=self.d_list.columns)
		for i in p_cnt[:-7].index:
			train_list = pd.concat([train_list,self.d_list[self.d_list['subject']==i]], axis=0)

		return train_list, val_list

	def cnt_image(self, image_list):
		classes_cnt = image_list['classname'].value_counts()
		ordered_idx = sorted(classes_cnt.index)
		c_cnt = classes_cnt[ordered_idx]
		c_per = round(classes_cnt[ordered_idx] / sum(classes_cnt) * 100, 1)
		classes_info = pd.DataFrame({'count': c_cnt, 'percent': c_per}, index=c_cnt.index)
		print('='*100)
		print(classes_info)


	def aug_image(self, image_list, fold, class_name, size):
		imgs = list()
		labels = list()

		c_dir = os.path.join(self.image_dir,class_name)
		imgs_dir = list(image_list[image_list['classname']==class_name]['img'])
		labels.extend(to_categorical([int(class_name[-1])]*len(imgs_dir), 10))
		for j in tqdm(range(len(imgs_dir))):
			im = load_img(c_dir + '/' + imgs_dir[j], target_size=size)
			arr = img_to_array(im)
			imgs.append(arr)

		return imgs, labels


	def save_test(self):
		test_dir = self.root_dir + '/dataset/test/'
		test_h5py = self.root_dir + '/dataset/h5py/test(224, 224, 3).h5'
		test_list = os.listdir(test_dir)
		print('='*100)
		print('Number of test images: ',len(test_list))
		

		rate = len(test_list) // 10

		test_imgs = os.listdir(test_dir) 
		test_imgs1 = test_imgs[:rate*1]
		test_imgs2 = test_imgs[rate*1:rate*2]
		test_imgs3 = test_imgs[rate*2:rate*3]
		test_imgs4 = test_imgs[rate*3:rate*4]
		test_imgs5 = test_imgs[rate*4:rate*5]
		test_imgs6 = test_imgs[rate*5:rate*6]
		test_imgs7 = test_imgs[rate*6:rate*7]
		test_imgs8 = test_imgs[rate*7:rate*8]
		test_imgs9 = test_imgs[rate*8:rate*9]
		test_imgs10 = test_imgs[rate*9:]
		test_imgs_list = [test_imgs1, test_imgs2, test_imgs3, test_imgs4, test_imgs5, test_imgs6, test_imgs7, test_imgs8, test_imgs9, test_imgs10]


		hf = h5py.File(test_h5py,'w')
		for i in range(len(test_imgs_list)):
			test_imgs_i = np.zeros((len(test_imgs_list[i]),224,224,3))
			for j in tqdm(range(len(test_imgs_list[i]))):
				im = load_img(test_dir + test_imgs_list[i][j], target_size=(224,224))
				arr = img_to_array(im)
				test_imgs_i[j] = arr
			hf.create_dataset('test' + str(i), data=test_imgs_i, compression='lzf')
		hf.close()


	def labeling(self, filename):
		print('='*100)
		print('Labeling test set')
		print('-'*100)
		filename = '11.submit'
		print('File name: ',filename)
		print()
		submit_dir = self.root_dir + '/result/concat_file/'
		submit_file = submit_dir + filename + '.csv'

		h5py_dir = self.root_dir + '/dataset/h5py/label_' + filename + '.h5'
		submit_h5py = h5py.File(h5py_dir, 'w')
		
		df = pd.read_csv(submit_file)
		print('Shape submit file: ',df.shape)
		c_df = df.iloc[:,1:]
		print('Shape submit file excluded img column: ',c_df.shape)

		label = np.argmax(np.array(c_df), axis=1)
		onehot = to_categorical(label, 10)

		submit_h5py.create_dataset('test_labels',data=onehot,compression='lzf')
		submit_h5py.close()



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--test',type=int,help="Save test images file")
	parser.add_argument('--labeling',default=False,type=bool, help="Whether try to 'self training' or not")
	parser.add_argument('--submit',default=None,type=str,help="submit file to labeling")
	args = parser.parse_args()

	P = Preprocessing()
	P.run(test=args.test, self_training=args.labeling, filename=args.submit)