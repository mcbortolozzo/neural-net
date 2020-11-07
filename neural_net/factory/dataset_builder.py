import numpy as np
import pandas as pd

from ..evaluation import StratifiedKFolds

class DatasetBuilder():

	@staticmethod
	def build_dataset_from_input_file(dataset_file):
		X = []
		Y = []

		try:
			with open(dataset_file, 'r') as f:
				for line in f.readlines():
					x_data, y_data = line.split(';')
					X.append([float(v) for v in x_data.split(',')])
					Y.append([float(v) for v in y_data.split(',')])

			return np.array(X), np.array(Y)
		except Exception as e:
			print(e)
			raise ValueError("Invalid dataset file")	

	@staticmethod
	def read_dataset_from_csv(dataset_file, target_column, separator='\t'):
		df = pd.read_csv(dataset_file, delimiter=separator)
		Y = df[[target_column]].values
		X = df.drop(target_column, axis=1).values
		return X, Y

	@staticmethod
	def read_dataset_from_csv_as_kfold(dataset_file, target_column, k, separator='\t'):
		df = pd.read_csv(dataset_file, delimiter=separator)
		feat_df = df.drop(target_column, axis=1)
		feat_df=(feat_df-feat_df.min())/(feat_df.max()-feat_df.min())
		target_df = df[[target_column]]
		df = pd.concat([feat_df, target_df], axis=1)
		kfolds = StratifiedKFolds(df, k, target_column)
		return kfolds