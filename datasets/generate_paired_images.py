import os
import os.path as path
from shutil import copyfile
import numpy as np
import glob
from nutsml import *
from nutsflow import *
from nutsflow.common import StableRandom
import matplotlib.pyplot as plt

diseases = ['Atelectasis',
						 'Cardiomegaly',
						 'Effusion',
						 'Infiltrate',
						 'Mass',
						 'Nodule',
						 'Pneumonia',
						 'Pneumothorax',
						 'Consolidation',
						 'Edema',
						 'Emphysema',
						 'Fibrosis',
						 'Pleural_Thickening',
						 'Hernia']



@nut_function
def Convert(sample, normal_as_extra=False):
	filename, labels, x, y, w, h = sample

	label_split = labels.split('|')

	if not normal_as_extra:
		label_vector = np.zeros(14, dtype=np.uint8)
		if labels != 'No Finding':
			for label in label_split:
				i = diseases.index(label)
				label_vector[i] = 1

	# normal as an extra label
	else:
		label_vector = np.zeros(14 + 1, dtype=np.uint8)
		if labels != 'No Finding':
			for label in label_split:
				i = diseases.index(label)
				label_vector[i] = 1
		else:
			label_vector[-1] = 1

	return filename, label_vector, int(x), int(y), int(w), int(h)


def get_file_labels(filepath, normal_as_extra=False):
	reader = ReadCSV(filepath, columns=(0, 1, 2, 3, 4, 5), skipheader=1)
	filenames_labels = (reader >> Convert(normal_as_extra) >> Collect())

	return filenames_labels


if __name__ == '__main__':
	# Folders read from
	# The csv file which contains disease labels and bounding box labels
	bbox_filepath = '/home/yunyan/Desktop/Pycharmprojects/MICCAI/Dataset7_3-MICCAI Localisation/BBox_List_2017.csv'
	# The folder contains all original images
	image_path = '/home/yunyan/Desktop/Pycharmprojects/MICCAI/attention-gans/data/images'

	# Folders to save bounding box only images, original images and labels.
	# Bounding box only images folder
	bbox_images = 'dataset/BBox/'
	# Folder to save the corresponding original images
	original_path = 'dataset/Images/'
	# This folder saves each bounding box location in a separate txt files.
	label_path = 'dataset/Labels/'

	# getting bounding box info
	filenames_labels = get_file_labels(bbox_filepath, normal_as_extra=False)


	for filename, label, x, y, w, h in filenames_labels:
		label_id = np.where(label == 1)[0][0]
		original_folder = path.join(original_path, str(label_id))
		if not path.exists(original_folder):
			os.makedirs(original_folder)
		bbox_folder = path.join(bbox_images, str(label_id))
		if not path.exists(bbox_folder):
			os.makedirs(bbox_folder)
		label_folder = path.join(label_path, str(label_id))
		if not path.exists(label_folder):
			os.makedirs(label_folder)

		src = path.join(image_path, filename)
		original_image = plt.imread(src)
		dst_ori = path.join(original_path, str(label_id), filename)
		plt.imsave(dst_ori, original_image, cmap='gray')
		original_copy = original_image.copy()
		original_image[y:y+h, x:x+w] = 0
		original_image = original_copy - original_image
		dst = path.join(bbox_images, str(label_id), filename)
		plt.imsave(dst, original_image, cmap='gray')

		txtpath = path.join(label_path, str(label_id), filename[:-3]+'txt')
		f = open(txtpath, 'a')
		f.write(str(label_id)+' '+str(x)+' '+str(y)+' '+str(w)+' '+str(h))
		f.close()

