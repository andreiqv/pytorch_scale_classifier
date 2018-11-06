#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
First run split_dataset.py to split dataset in train|valid|test parts.
Then run this script.

train loss=2.6459, acc=0.3861, top1=0.3861, top6=0.4985
valid loss=1.4961, acc=0.5917, top1=0.5847, top6=0.7120
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import math
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import copy

import progressbar
SHOW_BAR = True
DEBUG = False
TOPk = 6

import data_factory
import settings
data_dir = settings.data_dir 

#root = '/home/andrei/Data/Datasets/Scales/classifier_dataset_181018/'
#data_dir = '/w/WORK/ineru/06_scales/_dataset/splited/'

dataloaders, image_datasets = data_factory.load_data(data_dir)
#data_parts = list(dataloaders.keys())
dataset_sizes, class_names = data_factory.dataset_info(image_datasets)
num_classes = len(class_names)
data_parts = ['train', 'valid']

num_batch = dict()
num_batch['train'] = math.ceil(dataset_sizes['train'] / settings.batch_size)
num_batch['valid'] = math.ceil(dataset_sizes['valid'] / settings.batch_size)
print('train_num_batch:', num_batch['train'])
print('valid_num_batch:', num_batch['valid'])

#print(data_parts)
#print('train size:', dataset_sizes['train'])
#print('valid size:', dataset_sizes['valid'])
#print('classes:', class_names)
#print('class_to_idx:', dataset.class_to_idx)

#for i, (x, y) in enumerate(dataloaders['valid']):
#	print(x) # image
#	print(i, y) # image label

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        #res.append(correct_k.mul_(100.0 / batch_size))
        res.append(correct_k.mul_(1.0 / batch_size))
    
    return res


def accuracy_top1(outputs, labels):

	batch_size = len(outputs)
	res = np.zeros(batch_size, dtype=int)
	for i in range(batch_size):
		output = outputs[i].detach().cpu().numpy()
		label = int(labels[i])
		predict = np.argmax(output)
		res[i] = 1 if label==predict else 0
		if DEBUG: print('i={}: res={} (label={}, predict={})'.format(i, res[i], label, predict))
	return np.mean(res)


def accuracy_topk(outputs, labels, k=1):

	batch_size = len(outputs)
	res = np.zeros(batch_size, dtype=int)
	for i in range(batch_size):
		output = outputs[i].detach().cpu().numpy()
		label = int(labels[i])
		#predict = np.argmax(output)
		topk_predicts = set(output.argsort()[::-1][:k])
		res[i] = 1 if label in topk_predicts else 0
		if DEBUG: print('i={}: res={} (label={}, predicts={})'.format(i, res[i], label, topk_predicts))

	return np.mean(res)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('------------')
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in data_parts:
			if phase == 'train':
				scheduler.step()
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			if SHOW_BAR: bar = progressbar.ProgressBar(maxval=num_batch[phase]).start()

			# Iterate over data.
			acc1_list = []
			acc6_list = []

			for i_batch, (inputs, labels) in enumerate(dataloaders[phase]):
				inputs = inputs.to(device)
				labels = labels.to(device)

				if SHOW_BAR: bar.update(i_batch)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):

					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

					#print('outputs: ', outputs)
					#print('top-2: ', torch.topk(outputs, k=2))


				# statistics
				"""
				if DEBUG: print('----------')
				acc1, acc6 = accuracy(outputs, labels, topk=(1, TOPk))
				acc1 = acc1.double()
				acc6 = acc6.double()				
				if DEBUG: print('accuracy(1): acc1={}, acc6={}'.format(acc1, acc6))
				"""
				acc1 = accuracy_top1(outputs, labels)
				acc6 = accuracy_topk(outputs, labels, k=TOPk)
				if DEBUG: print('-')
				if DEBUG: print('accuracy(2): acc1={}, acc6={}'.format(acc1, acc6))

				acc1_list.append(acc1)
				acc6_list.append(acc6)
				if not SHOW_BAR:
					print('epoch {} [{}]: {}/{}'.format(epoch, phase, i_batch, num_batch[phase]))
					#print('preds: ', preds)
					#print('labels:', labels.data)
					print('match: ', int(torch.sum(preds == labels.data)))
					print('top1={:.4f}, top6={:.4f}'.format(acc1, acc6))

				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)				

			if SHOW_BAR: bar.finish()	

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]
			epoch_acc1 = np.mean(acc1_list)
			epoch_acc6 = np.mean(acc6_list)

			print('Epoch {} [{}]: loss={:.4f}, acc={:.4f}, top1={:.4f}, top6={:.4f}' .
				format(epoch, phase, epoch_loss, epoch_acc, epoch_acc1, epoch_acc6))

			# deep copy the model
			if phase == 'valid' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model



model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
	num_epochs=25)	