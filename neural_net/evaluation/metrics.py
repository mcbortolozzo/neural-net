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

	def accuracy(self):
		return self._TP().sum()/self.matrix.sum()

	def precision(self):
		pr = self._TP()/(self._TP() + self._FP())
		pr[np.isnan(pr)] = 1
		return pr

	def recall(self):
		re = self._TP()/(self._TP() + self._FN())
		re[np.isnan(re)] = 1
		return re

	def F1_score(self):
		return 2*self.precision()*self.recall()/(self.precision() + self.recall())

	def F1_macro(self):
		return self.F1_score().mean()

	def F1_micro(self):
		total_tp = self._TP().sum()
		total_fp = self._FP().sum()
		total_fn = self._FN().sum()
		micro_precision = total_tp/(total_tp+total_fp)
		micro_recall = total_tp/(total_tp+total_fn)
		return 2*micro_recall*micro_precision/(micro_precision+micro_recall)

	def _TP(self):
		return np.diag(self.matrix)

	def _FP(self):
		return self.matrix.sum(axis=1) - self._TP()

	def _FN(self):
		return self.matrix.sum(axis=0) - self._TP()


class RegressionMetrics():

	@staticmethod
	def rmse(actual, expected):
		return np.sqrt(np.sum(np.square(expected - actual))/len(actual))


