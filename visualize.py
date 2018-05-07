import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import pandas as pd
from .util import *

def slice_to_rgb(slice, number_to_index_dict, color_map):

	image = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
	for i in range(slice.shape[0]):
		for j in range(slice.shape[1]):
			if slice[i, j] == 0:
				continue
			if slice[i, j] in number_to_index_dict:
				row = number_to_index_dict[slice[i, j]]
				image[i, j, :] = color_map[row, :]

	return image

def show_slices(ul, ml, fsl, sub_ID, outpath=None):
	""" Function to display row of image slices """

	fig, axes = plt.subplots(3, 3, figsize=(8, 8))
	num_to_in, color_map = get_cmap()

	axes[0, 0].imshow(ul[:, :, 80], cmap="gray")
	axes[0, 1].imshow(ul[:, 128, :], cmap="gray")
	axes[0, 2].imshow(ul[128, :, :], cmap="gray")

	axes[1, 0].imshow(slice_to_rgb(ml[:, :, 80], num_to_in, color_map))
	axes[1, 1].imshow(slice_to_rgb(ml[:, 128, :], num_to_in, color_map))
	axes[1, 2].imshow(slice_to_rgb(ml[128, :, :], num_to_in, color_map))

	fsl = np.swapaxes(fsl[48:208,::-1, :], 2, 0)
	print(ul.shape, ml.shape, fsl.shape)
	axes[2, 0].imshow(slice_to_rgb(fsl[:, :, 80], num_to_in, color_map))
	axes[2, 1].imshow(slice_to_rgb(fsl[:, 128, :], num_to_in, color_map))
	axes[2, 2].imshow(slice_to_rgb(fsl[128,:,:], num_to_in, color_map))

	cols = ['Sagital', 'Axial', 'Coronal']
	rows = ['Unlabelled', 'Manually Labelled', 'Automatically Labelled']

	for ax, col in zip(axes[0], cols):
		ax.set_title(col)

	for ax, row in zip(axes[:, 0], rows):
		ax.set_ylabel(row, rotation=90)

	fig.suptitle('Subject Number ' + str(sub_ID))
	if outpath:
		plt.savefig(os.path.join(outpath,'raw_'+str(sub_ID)+'.pdf'))
	else:
		plt.show()

if __name__ == "__main__":

	data_path = ''
	sub_IDs = np.arange(2, 3)

	for sub_ID in sub_IDs:
		unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, data_path)
		uld = unlabelled.get_data()
		mld = man_labelled.get_data()
		fsld = FS_labelled.get_data()
		show_slices(uld, mld, fsld, sub_ID)