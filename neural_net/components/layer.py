import numpy as np

class Layer():

	def __init__(self, size, input_weights):
		self.size = size
		self.input_weights = np.array(input_weights)
		self.z_mem = None
		self.a_mem = None

	def activation(self, z):
		raise NotImplementedError()

	def get_squared_weights(self):
		#Exclude bias weights
		return self.input_weights[:, 1:] ** 2

	def forward(self, X):
		m = X.shape[0]
		X = np.hstack((np.ones((m, 1)), X))
		self.z_mem = np.matmul(self.input_weights,X.T).T
		self.a_mem = self.activation(self.z_mem)
		return self.a_mem


class SigmoidLayer(Layer):
	
	def __init__(self, size, input_weights):
		super().__init__(size, input_weights)

	def activation(self, z):
		return 1/(1 + np.exp(-z))
