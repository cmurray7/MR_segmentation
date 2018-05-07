import numpy as np
import nibabel as nib
import os
from imageio import imwrite
import pandas as pd
import matplotlib.pyplot as plt
from .util import *

def convert_to_2d(sub_IDs, outdir='tiff', un=True, man=True, fs=True):
	output_dir = outdir
	unlabelled_dir = 'unlabelled'
	labelled_dir = 'labelled'
	fs_labelled_dir = 'fs_labelled'
	data_path = ''

	for sub_ID in sub_IDs:
		unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, data_path)
		subject_dir = 'OASIS-TRT-20-' + str(sub_ID)

		if un:
			ud = unlabelled.get_data()
			uldir = os.path.join(output_dir, unlabelled_dir, subject_dir)
			if not os.path.exists(uldir):
				os.makedirs(uldir)

			for i in range(ud.shape[0]):
				imwrite(os.path.join(uldir, 'sag_slice_'+format(i, '03d')+'.tiff'), ud[i, :, :].astype(np.int16))

		if man:
			md = man_labelled.get_data()
			ldir = os.path.join(output_dir, labelled_dir, subject_dir)
			if not os.path.exists(ldir):
				os.makedirs(ldir)

			for i in range(md.shape[0]):
				imwrite(os.path.join(ldir, 'sag_slice_'+format(i, '03d')+'.tiff'), md[i, :, :].astype(np.int16))

		if fs:
			fld = FS_labelled.get_data()
			fldir = os.path.join(output_dir, fs_labelled_dir, subject_dir)
			if not os.path.exists(fldir):
				os.makedirs(fldir)

			for i in range(fld.shape[0]):
				imwrite(os.path.join(fldir, 'sag_slice_'+format(i, '03d')+'.tiff'), fld[i, :, :].astype(np.int16))

def convert_to_3d(sub_IDs, outdir='tiff3d'):
	unlabelled_dir = 'unlabelled'
	labelled_dir = 'labelled'
	fs_labelled_dir = 'fs_labelled'
	data_path = ''
	min_start = 35
	max_end = 230

	for sub_ID in sub_IDs:
		unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, data_path)
		subject_dir = 'OASIS-TRT-20-' + str(sub_ID)
		ud = unlabelled.get_data()
		mld = man_labelled.get_data()
		fld = FS_labelled.get_data()

		uldir = os.path.join(outdir, unlabelled_dir, subject_dir)
		if not os.path.exists(uldir):
			os.makedirs(uldir)

		mldir = os.path.join(outdir, labelled_dir, subject_dir)
		if not os.path.exists(mldir):
			os.makedirs(mldir)

		fldir = os.path.join(outdir, fs_labelled_dir, subject_dir)
		if not os.path.exists(fldir):
			os.makedirs(fldir)

		for i in range(1, ud.shape[2]-1):
			imwrite(os.path.join(uldir, 'sag_3d_'+format(i, '03d')+'.tiff'),
			        ud[:, min_start:max_end, i-1:i+2].astype(np.int16))

			imwrite(os.path.join(fldir, 'sag_3d_'+format(i, '03d')+'.tiff'),
			        fld[:, min_start:max_end, i].astype(np.int16))

			imwrite(os.path.join(mldir, 'sag_3d_' + format(i, '03d') + '.tiff'),
			        mld[:, min_start:max_end, i].astype(np.int16))

def convert_to_pickled(sub_IDs, number_to_index_dict, outdir='pickled'):

	unlabelled_dir = 'unlabelled'
	labelled_dir = 'labelled'
	min_start = 35
	max_end = 230

	for sub_ID in sub_IDs:
		unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, '')

		subject_path = 'OASIS-TRT-20-' + str(sub_ID)
		ud = unlabelled.get_data()
		out_ud = ud[:,min_start:max_end, :]/np.max(ud)
		np.save(os.path.join(outdir, unlabelled_dir, subject_path), out_ud.astype(np.float32))

		mld = man_labelled.get_data()
		mld = mld[:, min_start:max_end, :]

		for i in range(mld.shape[0]):
			for j in range(mld.shape[1]):
				for k in range(mld.shape[2]):
					if mld[i, j, k] == 0:
						pass
					elif mld[i,j,k] in number_to_index_dict:
						mld[i,j,k] = number_to_index_dict[mld[i,j,k]]
		print(np.min(mld), np.max(mld))
		np.save(os.path.join(outdir, labelled_dir, subject_path), mld.astype(np.int64))

if __name__ == "__main__":
	output_dir = 'tiff'
	unlabelled_dir = 'unlabelled'
	data_path = ''

	n2id, cmap = get_cmap()
	sub_ids = range(1, 21)
	convert_to_pickled(sub_ids, n2id)




	# min_fw = 256
	# max_bw = 0
	# for sub_ID in range(1,21):
	# 	print("Subject #: ", sub_ID)
	# 	unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, data_path)
	#
	# 	ud = unlabelled.get_data()
	# 	print("Unlabeled data type : ", ud.dtype)
	# 	nonzero = np.count_nonzero(ud, axis=(0,2))
	# 	forward = np.argmax(nonzero!=0)
	# 	backward = len(nonzero) - np.argmax(nonzero[::-1]!=0)
	#
	# 	if forward < min_fw:
	# 		min_fw = forward
	#
	# 	if max_bw < backward:
	# 		max_bw = backward
	#
	# 	# fd = FS_labelled.get_data()
	# 	# print("FS labeled data type : ", fd.dtype)
	# 	# print("min: ", np.min(fd))
	# 	# print("max: ", np.max(fd))
	# 	#
	# 	# md = man_labelled.get_data()
	# 	# print("Manually labeled data type : ", md.dtype)
	# 	# print("min: ", np.min(md))
	# 	# print("max: ", np.max(md))