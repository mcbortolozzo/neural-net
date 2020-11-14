import numpy as np
import pandas as pd
import json
import itertools

from neural_net.factory import NetworkBuilder, DatasetBuilder
from neural_net.evaluation import DataLoader, ConfusionMatrix, RegressionMetrics
from neural_net.components import MinMaxScaler

EPSILON = 0.000001
EPOCHS = 1000
VERBOSE = False

DATASETS = [ 
	 {'file': 'data/wine_recognition.tsv', 'target': 'target', 'onehot': True, 'regression': False, 'experiment_specs': 'data/experiment_specs_wine.json'},
	 {'file': 'data/house_votes_84.tsv', 'target': 'target', 'onehot': False, 'regression': False, 'experiment_specs': 'data/experiment_specs_votes.json'},
	 {'file': 'data/houses_to_rent_cleaned.tsv', 'target': 'target', 'onehot': False, 'regression': True, 'experiment_specs': 'data/experiment_specs_houses_2.json'}
]

OUTPUT_FILE = 'wine_recognition_loss.json'

def run_experiment(dataset_file, target_column, specs, onehot=False, regression=False, verbose=False, check_gradient=False):

	lr = specs[1]
	batch_size = specs[2]
	k_count = specs[3]
	lambd = specs[4]
	network_specs = specs[0]
	epochs = EPOCHS

	print('========== Starting Experiment ===========')
	print('Learning Rate: \t\t %.5f' % lr)
	print('Batch Size: \t\t %d' % batch_size)
	print('K-Folds: \t\t %d' % k_count)
	print('Lambda: \t\t %.4f' % lambd)
	print('Netowrk Architecture: \t %s' % "IN " +"-".join([str(x['size']) for x in network_specs]) + " OUT")

	kfolds = DatasetBuilder.read_dataset_from_csv_as_kfold(dataset_file, target_column, k_count)

	results = {'specs': specs, 'folds': []}

	for fold_idx, (train, test) in enumerate(kfolds.get_folds()):
		print('------ Starting Fold %d ----------' % (fold_idx+1))
		fold_results = {'epochs': {}}
		results['folds'].append(fold_results)

		nnet = NetworkBuilder.build_network_from_specs(lambd, lr, network_specs, regression)
		dataloader = DataLoader(train.drop(target_column, axis=1).values, train[[target_column]].values, batch_size, shuffle=True, onehot=onehot)
		prev_loss = 99999999

		loss = 0
		loss_count = 0
		for epoch in range(epochs):
			for data in dataloader:
				X, Y = data	
				pred = nnet.forward(X)
				loss += nnet.loss(pred, Y)
				loss_count += 1
				nnet.backprop(Y)


			loss /= loss_count
			if(verbose):
				print("Epoch: %d \t Loss: %f" % (epoch, loss))		
				fold_results['epochs'][epoch] = {
					'loss': loss,
				}

			if abs(loss - prev_loss) < EPSILON:
				if verbose:
					print('stopped because of small loss gain')
				break
			else:
				prev_loss = loss
				loss = 0
				loss_count = 0

		X_test = test.drop(target_column, axis=1).values
		Y_test = test[[target_column]].values
		Y_pred = nnet.forward(X_test)

		if not regression:			
			cf_matrix = ConfusionMatrix(train[target_column].max()+1)
			cf_matrix.update(Y_pred, Y_test, onehot)

			fold_results['cf_matrix'] = cf_matrix.matrix.tolist()
			fold_results['accuracy'] = cf_matrix.accuracy()
			fold_results['f1_macro'] = cf_matrix.F1_macro()
		else:
			fold_results['rmse'] = RegressionMetrics.rmse(Y_pred, Y_test)
			fold_results['mse'] = RegressionMetrics.mse(Y_pred, Y_test)
			fold_results['mean_error'] = RegressionMetrics.mean_error(Y_pred, Y_test)
			print(fold_results)
			

	if not regression:
		results['accuracy'] = sum([f['accuracy'] for f in results['folds']])/len(results['folds'])
		results['f1_macro'] = sum([f['f1_macro'] for f in results['folds']])/len(results['folds'])
	else:
		results['rmse'] = sum([f['rmse'] for f in results['folds']])/len(results['folds'])


	return results


full_results = {'datasets': {}}


for data in DATASETS:
	print("########### Dataset %s ###########" % data['file'])
	full_results['datasets'][data['file']] = {'best_results': None, 'experiments' : []}
	current_dataset_results = full_results['datasets'][data['file']]
	best_accuracy = 0
	best_specs = None

	with open(data['experiment_specs'], 'r') as f:
		experiments = json.load(f)
		experiment_iterations = itertools.product(*experiments.values())

		for exp_specs in experiment_iterations:
			with np.errstate(invalid='ignore'):
				results = run_experiment(data['file'], data['target'], exp_specs, data['onehot'], data['regression'], VERBOSE)
			current_dataset_results['experiments'].append({'specs': exp_specs, 'results': results})
			if data['regression']:
				if results['rmse'] <= best_accuracy:
					print('New best rmse:', results['rmse'])
					best_accuracy = results['rmse']
					best_specs = exp_specs
				else:
					print("rmse: %.4f" % results['rmse'])
			else:
				if results['accuracy'] >= best_accuracy:
					print('New best accuracy:', results['accuracy'])
					best_accuracy = results['accuracy']
					best_specs = exp_specs
				else:
					print("Accuracy: %.4f" % results['accuracy'])

		if data['regression']:
			current_dataset_results['best_results'] = {'specs': best_specs, 'rmse': best_accuracy}
		else:
			current_dataset_results['best_results'] = {'specs': best_specs, 'accuracy': best_accuracy}

with open(OUTPUT_FILE, 'w') as f:
	json.dump(full_results, f)