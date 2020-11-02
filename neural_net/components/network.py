import numpy as np

class NeuralNet():

	def __init__(self, lambd):
		self.lambd = lambd
		self.lr = 0.1
		self.layers = []

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward(self, input_activation):
		activation = input_activation
		for l in self.layers:
			activation = l.forward(activation)

		return activation

	def calculate_regularization(self, m):
		S = 0
		for l in self.layers:
			S += np.sum(l.get_squared_weights())

		S *= self.lambd/(2*m)
		return S

	def loss(self, x_pred, Y, should_reduce=True):
		m = Y.shape[0]
		loss = np.sum(-Y* np.log(x_pred) - (1-Y)*np.log(1-x_pred), axis=1)
		loss = np.mean(loss)
		regularization = self.calculate_regularization(m)
		return loss + regularization

	def backprop(self, Y):
		d = self.layers[-1].a_mem - Y
		for l in reversed(self.layers):
			d = l.backprop(d, self.lambd, self.lr)