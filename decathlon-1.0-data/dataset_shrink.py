import numpy as np
import os
import glob
import argparse
import random

parser = argparse.ArgumentParser('Dataset shrinking')

parser.add_argument('--save-dir', dest='save_dir', type=str)
parser.set_defaults(shrink_dir='decathlon-1.0-data-tenth')

parser.add_argument('--shrink-ratio', dest='shrink_ratio', type=int)
parser.set_defaults(shrink_ratio=10)

args = parser.parse_args()

data_name = ['cifar100', 'aircraft', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot',
			 'svhn', 'ucf101', 'vgg-flowers', 'cifar10', 'caltech256', 'sketches']

prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dirname = args.save_dir

if not os.path.exists(os.path.join(prj_dir, dirname)):
	os.mkdir(os.path.join(prj_dir, dirname))


for i in range(len(data_name)):
	dataset = data_name[i]
	if not os.path.exists(os.path.join(prj_dir, dirname, dataset)):
		os.mkdir(os.path.join(prj_dir, dirname, dataset))
	sel_num = 0
	for mode in ['train']:
		cate_list = glob.glob('./{}/{}/*'.format(dataset, mode))
		for j in range(len(cate_list)):
			if dataset == 'sketches':
				file_ext = '/*.png'
			else:
				file_ext = '/*.jpg'
			img_list = glob.glob(cate_list[j] + file_ext)
			random.seed(i*len(data_name)+j)
			random.shuffle(img_list)
			subfolder = cate_list[j].split('/', 3)[3]
			
			if not os.path.exists(os.path.join(prj_dir, dirname, dataset, mode)):
				os.mkdir(os.path.join(prj_dir, dirname, dataset, mode))
			if not os.path.exists(os.path.join(prj_dir, dirname, dataset, mode, subfolder)):
				os.mkdir(os.path.join(prj_dir, dirname, dataset, mode, subfolder))

			one_sel = len(img_list) // args.shrink_ratio
			sel_num += one_sel
			for k in range(one_sel):
				picName = os.path.join(prj_dir, dirname, dataset, mode, subfolder, os.path.basename(img_list[k]))
				os.system('cp {} {}'.format(img_list[k], picName))

	# val
	os.system('cp -r {} {}'.format(os.path.join(dataset, 'val'), os.path.join(prj_dir, dirname, dataset)))
	# test
	os.system('cp -r {} {}'.format(os.path.join(dataset, 'test'), os.path.join(prj_dir, dirname, dataset)))

	print('{} train:{}'.format(dataset, sel_num))
