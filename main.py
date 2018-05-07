import sys
sys.path.insert(0, "~/Documents/spring2018/543/project/code/")
import numpy as np
from sklearn.metrics import f1_score

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from oasis_data import OasisDataset
from MSD_cnn import MSDNet
from util import *
#
NUM_CLASSES = 67
BATCH_SIZE = 25

def val_f1(net, val_loader):

	f1_sum = 0
	for val_data, val_label in val_loader:
		start = np.random.randint(45, 215)
		for j in range(start, start+50):
			output = net(val_data[:, j-2:j+3, :, :])
			output = from_one_hot(output)
			curr_f1 = f1_score(val_label[:, j, :, :].view(-1), output.view(-1),
			                   labels=np.unique(val_label)[1:], average='weighted')
			f1_sum += curr_f1
	return f1_sum/50


def train(net, train_loader, val_loader, epochs, save=False):

	save_epoch = 5

	weights = np.ones(NUM_CLASSES, dtype=np.float32)
	weights[0] = 0.000665944471330417
	loss_func = nn.CrossEntropyLoss()

	optimizer = optim.Adam(net.parameters(), lr = 0.01)

	losses = np.zeros(10000)
	all_F1s = []
	iter_ = 0

	for e in range(1, epochs+1):
		net.train()

		for batch_idx, (data, label) in enumerate(train_loader):
			data, label = Variable(data), Variable(label)
			output = net(data)
			loss = loss_func(output, label)
			loss.backward()
			optimizer.step()

			losses[iter_] = loss.data.item()


		f1 = val_f1(net, val_loader)
		all_F1s.append(f1)
		print("Epoch {}/{} |\t Loss: {} |\t F1: {}".format(e, epochs, losses[iter_], f1))

		if e % save_epoch == 0 and save:
			torch.save(net.state_dict(), './current_model_epoch_'+str(e))

	torch.save(net.state_dict(), './final_model_'+str(epochs)+'_epochs')
	np.save('./loss_10_epochs', losses)
	np.save('./acc_10_epochs', np.asarray(all_F1s))

	return net

def test(net, test_loader):
	net.load_state_dict(torch.load('./final_model_10_epochs'))
	for test_data, test_label in test_loader:
		print(test_data.shape, test_label.shape)
	net.eval()
	# for each subject
		# for each slice
			# label slice
		#save volume as pickle



def main(task, save):

	net = MSDNet()
	train_set = OasisDataset('train')
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

	val_set = OasisDataset('val')
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=40)

	test_set = OasisDataset('test')
	test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

	if task == 'train':
		trained_net = train(net, train_loader, val_loader, 10, save=save)
	elif task == 'test':
		test(net, test_loader)


if __name__ == "__main__":
	task = sys.argv[1]
	# Pass save parameter as 0 (False) or 1 (True)
	save = sys.argv[2]
	main(task, save)