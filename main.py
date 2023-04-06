from prepare import save_h5
from train import train
from test import test

if __name__ == '__main__':
	# scale:  Magnification
	scale = 3
	# patch_size: size you want to cut the images into pieces for preprocessing and training
	patch_size = 84
	# stride: stride when cutting the images
	stride = 29
	# cache_size: equal to training batch_size
	cache_size=256
	print('data prepare phase')
	# phase: 'train' or 'valid'
	# batch_size: depanding on your RAM size, should be a factor of your train/valid images number. For example, you got 800 train images, batch_size could be 100
	save_h5(imgs_dir=r'./Datasets/DIV2K_train_HR', phase='train', scale=scale, patch_size=patch_size, stride=stride, batch_size=100, cache_size=cache_size)
	print('data prepare phase done')
	print('train phase')
	# model_name: 'OFSRCNN' or 'FSRCNN'
	# num_epochs: number of epochs
	# continue_epoch: how many epochs you have trained when continue training
	train(scale=scale, patch_size=patch_size, model_name='OFSRCNN', num_epochs=100, continue_epoch=0, batch_size=cache_size)
	print('train phase done')
	print('test phase')
	# model_name: 'OFSRCNN' or 'FSRCNN'
	# dataset_name: 'Set5' or 'Set14'
	test(scale=scale, model_name='OFSRCNN', dataset_name='Set5')
	print('test phase done')