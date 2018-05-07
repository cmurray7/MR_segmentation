import numpy as np
import nibabel as nib
import os
import pandas as pd

NUM_CLASSES = 69
BATCH_SIZE = 10

def to_one_hot(label_slice):
	return (np.arange(NUM_CLASSES) == label_slice[...,None]).astype(np.int64)

def from_one_hot(label_volume):
	print(label_volume.shape)
	return(np.argmax(label_volume, axis=-1))

def subject_volumes(subject_ID, data_path):

	raw_dir = 'OASIS-TRT-20_volumes'
	subject_dir = 'OASIS-TRT-20-' + str(subject_ID)
	FS_folder = 'processed_FreeSurfer'

	unlabelled_fn = 't1weighted_brain.nii.gz'
	manually_labelled_fn = 'labels.DKT31.manual.nii.gz'
	FS_labelled_fn = 'NMM_FINAL_LABELED_FILES/aparcNMMjt+aseg.nii.gz'

	unlabelled = nib.load(os.path.join(data_path, raw_dir, subject_dir, unlabelled_fn))
	man_labelled = nib.load(os.path.join(data_path, raw_dir, subject_dir, manually_labelled_fn))
	FS_labelled = nib.load(os.path.join(data_path, FS_folder, subject_dir, FS_labelled_fn))

	return unlabelled, man_labelled, FS_labelled

def get_cmap():
	all_info = pd.read_table('code/FreeSurferCortical.txt', sep='\s+', comment='#',
	              names=['number', 'label', 'R', 'G', 'B', 'A'],
	              dtype={'number':np.float32, 'label':object, 'R':np.uint8, 'G':np.uint8, 'B':np.uint8, 'A':np.uint8})

	number_to_index_dict = pd.Series(all_info.index.values, index=all_info.number).to_dict()

	return number_to_index_dict, np.asarray(all_info.loc[:, 'R':'B'])

def slice_to_rgb(slice, number_to_index_dict, color_map):

	image = np.zeros((slice.shape[0], slice.shape[1], 3), dtype=np.uint8)
	for i in range(slice.shape[0]):
		for j in range(slice.shape[1]):
			if slice[i, j] == 0:
				continue
			row = number_to_index_dict[slice[i, j]]
			image[i, j, :] = color_map[row , :]
	return image