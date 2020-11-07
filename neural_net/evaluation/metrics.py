import numpy as np
import pandas as pd

class ConfusionMatrix():

	def __init__(self, class_count):
		self.matrix = np.zeros((class_count, class_count))

	def update(self, y_actual, y_expected, onehot):
		if onehot:			
			pred = np.zeros_like(y_actual)
			pred[np.arange(len(y_actual)), y_actual.argmax(1)] = 1
			pred = pd.Series(np.where(pred == 1)[1].reshape(len(y_actual)))
		else:
			pred = pd.Series(np.around(y_actual).astype(int).reshape(len(y_actual)))

		y_expected = pd.Series(y_expected.reshape(len(y_actual)))

		for i in range(len(y_expected)):
			
			self.matrix[pred[i]][y_expected[i]] += 1


