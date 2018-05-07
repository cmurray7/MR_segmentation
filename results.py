import numpy as np
import matplotlib.pyplot as plt
import imageio
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

import os
import sys
sys.path.insert(0, "~/Documents/spring2018/543/project/code/")
from util import slice_to_rgb, subject_volumes, get_cmap


def comparison_slices(fs, sc, actual, number_to_index_dict, cmap, id):

	actual_im = slice_to_rgb(actual, number_to_index_dict, cmap)
	fs_im = slice_to_rgb(fs, number_to_index_dict, cmap)
	sc_im = slice_to_rgb(sc, number_to_index_dict, cmap)

	fig, axes = plt.subplots(1, 3)
	axes[0].imshow(actual_im)
	axes[0].set_title('Manual Labelling')
	axes[1].imshow(fs_im)
	axes[1].set_title('Free Surfer Labelling')
	axes[2].imshow(sc_im)
	axes[2].set_title('MS-D CNN Labelling')

	plt.savefig('outfiles/'+id+'.pdf')

def predictions_to_labels(output, p2n):

	holder = np.zeros(output.shape)
	for i in range(output.shape[0]):
		for j in range(output.shape[1]):
			holder[i, j] = p2n[output[i, j]]

	return holder

def performance(actual, predicted):
	acc = accuracy_score(actual.ravel(), predicted.ravel())
	prec = precision_score(actual.ravel(), predicted.ravel(), average='weighted', labels=np.unique(actual)[1:])
	rec = recall_score(actual.ravel(), predicted.ravel(), average='weighted', labels=np.unique(actual)[1:])
	return np.array([acc, prec, rec])

if __name__ == "__main__":

	slide_cam_labelled_path = 'tiff/slide_cam_2_extra_slices/OASIS-TRT-20-'
	n2id, cmap = get_cmap()
	id2n = {v: k for k, v in n2id.items()}
	print(len(n2id))
	test_sub_IDs = range(1, 21)

	fs_perf = np.zeros((len(test_sub_IDs), 3))
	sc_perf = np.zeros((len(test_sub_IDs), 3))

	for j, sub_ID in enumerate(test_sub_IDs):
		unlabelled, man_labelled, FS_labelled = subject_volumes(sub_ID, '')
		uld = unlabelled.get_data()
		mld = man_labelled.get_data()

		print(len(np.unique(mld)))

# 		fsld = FS_labelled.get_data()
# 		fsld = np.swapaxes(fsld[48:208,::-1, :], 2, 0)
# 		fsld[np.where(fsld<=1000)] = 0
#
# 		scl = np.empty((256, 256, 160))
# 		i = 0
# 		for scpath in sorted(os.listdir(slide_cam_labelled_path+str(sub_ID))):
# 			curr_path = slide_cam_labelled_path + str(sub_ID) + '/' + scpath
# 			scl[i, :, :] = predictions_to_labels(imageio.imread(curr_path), id2n)
# 			i += 1
#
# 		print('subject ', sub_ID)
# 		print(np.unique(mld))
# 		print(np.unique(fsld))
# 		print(np.unique(scl))
#
# 		fs_perf[j, :] = performance(mld, fsld)
# 		print(fs_perf[j, :])
# 		sc_perf[j, :] = performance(mld, scl)
# 		print(sc_perf[j, :])
#
# 		plt.figure()
# 		cm = confusion_matrix(mld.ravel(), scl.ravel())
# 		np.fill_diagonal(cm, 0)
# 		plt.imshow(cm, interpolation='nearest')
# 		plt.title('Segment Confusion for Subject '+ str(sub_ID) + '(omitting diagonal)')
# 		plt.colorbar()
# 		plt.savefig('outfiles/msdcnn_confusion_sub_'+str(sub_ID)+'.pdf')
#
# 		fn = 'sub_'+str(sub_ID)+'_comparison_2d'
# 		comparison_slices(fsld[128, :, :], scl[128, :, :], mld[128,:,:], n2id, cmap, fn)
#
# np.savetxt('outfiles/fs_2d_res.csv',fs_perf,delimiter=',', header = 'Accuracy,Precision,Recall')
# np.savetxt('outfiles/sc_2d_res.csv',sc_perf,delimiter=',', header = 'Accuracy,Precision,Recall')