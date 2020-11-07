import numpy as np

class DataLoader():

	def __init__(self, X, Y, batch_size, shuffle, onehot):
		self.X = X
		if onehot:
			self.Y = self.encode_onehot(Y)
		else:
			self.Y = Y
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __iter__(self):
		return DataLoaderIterator(self)

	def encode_onehot(self, Y):
		n_values = Y.max()+1
		encoded_Y = np.eye(n_values)[Y].reshape(len(Y), n_values)
		return encoded_Y
		
	def get_data(self):
		if self.shuffle:
			p = np.random.permutation(len(self.X))
			return self.X[p], self.Y[p]
		else:
			return self.X, self.Y

class DataLoaderIterator():

	def __init__(self, dataloader):
		self._X, self._Y = dataloader.get_data()
		self._batch_size = dataloader.batch_size
		self._index = 0

	def __next__(self):
		if self._index >= len(self._X):
			raise StopIteration
		
		end_idx = min(self._index + self._batch_size, len(self._X))
		result = (self._X[self._index: end_idx, :], self._Y[self._index: end_idx, :])
		self._index = end_idx
		return result

		

