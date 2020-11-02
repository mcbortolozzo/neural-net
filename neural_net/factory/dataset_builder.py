import numpy as np

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