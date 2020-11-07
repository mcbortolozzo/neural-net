import numpy as np
import pandas as pd

from neural_net.factory import NetworkBuilder, DatasetBuilder
from neural_net.evaluation import DataLoader, ConfusionMatrix

np.random.seed(1)

TEST_SPECS = { 'learning_rate': 0.1, 'lambda': 0.1, 'kfolds': 10, 'network_specs':[{'size': 13, 'type': 'sigmoid'}, {'size': 4, 'type': 'sigmoid'}]}

K = 10
BATCH_SIZE = 100
EPOCHS = 1000
EPSILON = 0.00000001

def run_experiment(dataset_file, target_column, specs, onehot = False, check_gradient = False):
	lr = specs['learning_rate']
	lambd = specs['lambda']
	network_specs = specs['network_specs']
	k_count = specs['kfolds']

	kfolds = DatasetBuilder.read_dataset_from_csv_as_kfold(dataset_file, target_column, k_count)

	for train, test in kfolds.get_folds():
		nnet = NetworkBuilder.build_network_from_specs(lambd, lr, network_specs)
		cf_matrix = ConfusionMatrix(train[target_column].max()+1)
		dataloader = DataLoader(train.drop(target_column, axis=1).values, train[[target_column]].values, BATCH_SIZE, shuffle=True, onehot=onehot)

		for epoch in range(EPOCHS):
			i = 0
			for data in dataloader:
				X, Y = data	
				pred = nnet.forward(X)
				# print('X', X)
				# print('pred', pred)
				# print('Y', Y)
				loss = nnet.loss(pred, Y)
				# print('W', nnet.layers[0].W)
				nnet.backprop(Y)
				# print('W', nnet.layers[0].W)
				i+=1
				# if epoch == 10:
				# 	exit()

				if check_gradient:
					grad_check = nnet.verify_gradient(X,Y,EPSILON)
					for i, l in enumerate(nnet.layers):
						print(np.max(abs(grad_check[i] - l.grad_mem)))
						assert np.less(abs(grad_check[i] - l.grad_mem), 10e-4).all(), (grad_check[i], l.grad_mem)



			if epoch % 100 == 0:
				print("curent loss %.5f" % loss)

		X_test = test.drop(target_column, axis=1).values
		Y_test = test[[target_column]].values
		Y_pred = nnet.forward(X_test)

		# print(Y_pred, Y_test)

		cf_matrix.update(Y_pred, Y_test, onehot)

		print(cf_matrix.matrix)



run_experiment('data/test/wine_recognition.tsv', 'target', TEST_SPECS, True, False)