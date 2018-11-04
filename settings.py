#export CUDA_VISIBLE_DEVICES=3
import os

if os.path.exists('.local'):
	data_dir = '/w/WORK/ineru/06_scales/_dataset/splited/'
	batch_size = 4
	num_workers = 4

else:
	data_dir = '/home/andrei/Data/Datasets/Scales/splited/'
	batch_size = 32
	num_workers = 8	