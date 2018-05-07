import sys
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable

from .oasis_data import OasisDataset
from .MSD_cnn import MSDNet

NUM_CLASSES = 69
BATCH_SIZE = 10

def CrossEntropy2d(inp, target, weight=None, size_average=True):
	""" 2D version of the cross entropy loss """
	# print inp.size()
	# print target.size()
	dim = inp.dim()
	if dim == 2:
		return F.cross_entropy(inp, target, weight, size_average)
	elif dim == 4:
		output = inp.view(inp.size(0),inp.size(1), -1)
		output = torch.transpose(output,1,2).contiguous()
		output = output.view(-1,output.size(2))
		target = target.view(-1)
		return F.cross_entropy(output, target,weight, size_average)
	else:
		raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def calculate_val_accuracy(net, val_loader):
	correct_nonzero = 0.0
	total_nonzero = 1.0
	for val_data, val_label in val_loader:

		output = torch.max(net(val_data), 1)[1]

		total_nonzero += np.count_nonzero(val_label)
		correct_nonzero += np.count_nonzero(np.logical_and(val_label!=0, val_label==output))

	return correct_nonzero*100/total_nonzero

def train(net, train_loader, val_loader, epochs):

	save_epoch = 5

	weights = np.ones(NUM_CLASSES, dtype=np.float32)
	weights[0] = 0.000665944471330417
	# loss_func = nn.CrossEntropy2d(weight=torch.Tensor(weights))

	optimizer = optim.Adam(net.parameters(), lr = 0.0001)

	losses = np.zeros(1000000)
	mean_losses = np.zeros(100000000)
	iter_ = 0

	all_losses = []
	all_accs = []

	for e in range(1, epochs+1):
		net.train()

		epoch_losses = []
		epoch_accs = []

		for batch_idx, (data, label) in enumerate(train_loader):
			data, label = Variable(data), Variable(label)
			output = net(data)
			loss = CrossEntropy2d(output, label, weight=torch.Tensor(weights), size_average=True)
			loss.backward()
			optimizer.step()

			epoch_losses.append(loss[0])
			val_acc = calculate_val_accuracy(net, val_loader)
			epoch_accs.append(val_acc)

			losses[iter_] = loss.data[0]
			# mean_losses[iter_] = np.mean(losses[max(0, iter_ -100):iter])

			if iter_ % 100 == 0:
				print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
					e, epochs, batch_idx, 10,
					100. * batch_idx / 10, sum(epoch_losses)/len(epoch_losses), sum(epoch_accs)/len(epoch_accs)))
			iter_ += 1

		current_epoch_loss = sum(epoch_losses)/len(epoch_losses)
		current_epoch_acc = sum(epoch_accs)/len(epoch_accs)

		all_losses.append(current_epoch_loss)
		all_accs.append(current_epoch_acc)

		"Epoch {}/{} |\t Loss: {} |\t Accuracy: {}".format(e, epochs, current_epoch_loss, current_epoch_acc)

		if e % save_epoch == 0:
			torch.save(net.state_dict(), './current_model_epoch_'+str(e))

	torch.save(net.state_dict(), './final_model_'+str(e)+'_epochs')
	np.save('./loss_10_epochs', np.asarray(all_losses))
	np.save('./acc_10_epochs', np.asarray(all_accs))

	return net

def test(net, test_loader):
	net.load_state_dict(torch.load('./final_model_10_epochs'))



def main(task):

	net = MSDNet()
	train_set = OasisDataset('train')
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)

	val_set = OasisDataset('val')
	val_loader = torch.utils.data.DataLoader(val_set, batch_size=51, shuffle=False)

	test_set = OasisDataset('test')
	test_loader = torch.utils.data.DataLoader(test_set, shuffle=False)

	if task == 'train':
		trained_net = train(net, train_loader, val_loader, 10)
	elif task == 'test':
		test(net, test_loader)


if __name__ == "__main__":
	task = sys.argv[1]
	main(task)
