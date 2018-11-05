import os
import sys
import random


#src_dir = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'
#dst_dir = '/home/andrei/Data/Datasets/Scales/splited/'

src_dir = '/w/WORK/ineru/06_scales/_dataset/copy/'
dst_dir = '/w/WORK/ineru/06_scales/_dataset/splited/'
parts = ['train', 'valid', 'test']


def copy_files_to_subdirs(src_dir, dst_dir, parts, ratio=[9,1,0]):

	src_dir = src_dir.rstrip('/')
	dst_dir = dst_dir.rstrip('/')

	os.system('mkdir -p {}'.format(dst_dir))	
	for p in parts:
		os.system('mkdir -p {}'.format(dst_dir + '/' + p))

	subdirs = os.listdir(src_dir)
	for class_name in subdirs:
		subdir = src_dir + '/' + class_name
		if not os.path.isdir(subdir): continue

		file_names = os.listdir(subdir)
		if len(file_names) == 0: 
			print('{0} - empty subdir'.format(class_name))
			continue
		
		# calculate train, valid and test sizes
		num_files = len(file_names)
		num_valid = num_files * ratio[1] // sum(ratio)
		num_test  = num_files * ratio[2] // sum(ratio)
		num_train = num_files - num_valid - num_test

		min_num_train = 10
		if num_train < min_num_train:
			(num_train, num_valid, num_test) = (num_files, 0, 0)

		#random.shuffle(file_names)
		file_names.sort()
		files = dict()
		files['train'] = file_names[ : num_train]
		files['valid'] = file_names[num_train : num_train + num_valid]
		files['test']  = file_names[num_train + num_valid : ]
		print('[{}] - {} - {}:{}:{}'.\
			format(class_name, num_files, num_train, num_valid, num_test))
		#print('train:valid:test = ', len(files['train']),\
		#	len(files['valid']), len(files['test']))

		for part in parts:
			cmd = 'mkdir -p {}'.format(dst_dir + '/' + part + '/' + class_name)
			os.system(cmd)
			#print(cmd)
			for file_name in files[part]:
				src_path = subdir + '/' + file_name
				dst_path = dst_dir + '/' + part + '/' + class_name + '/' + file_name
				cmd = 'cp {} {}'.format(src_path, dst_path)
				os.system(cmd)
				#print(cmd)

if __name__ == '__main__':

	copy_files_to_subdirs(src_dir, dst_dir, parts, ratio=[9,1,0])

