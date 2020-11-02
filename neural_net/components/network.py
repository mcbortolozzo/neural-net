from .layer import SigmoidLayer

import numpy as np
import copy

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

	def loss(self, x_pred, Y):
		m = Y.shape[0]
		loss = np.sum(-Y* np.log(x_pred) - (1-Y)*np.log(1-x_pred), axis=1)
		loss = np.mean(loss)
		regularization = self.calculate_regularization(m)
		return loss + regularization

	def backprop(self, Y):
		d = self.layers[-1].a_mem - Y
		for l in reversed(self.layers):
			d = l.backprop(d, self.lambd, self.lr)

	def get_loss_for_grad_check(self, X, Y, epsilon, layer, layer_idx, i, j):
		self.layers[layer_idx].W[i, j] = self.layers[layer_idx].W[i, j] + epsilon
		x_pred = self.forward(X)
		loss = self.loss(x_pred, Y)
		self.layers[layer_idx].W[i, j] = self.layers[layer_idx].W[i, j] - epsilon	

		return loss

	def verify_gradient(self, X, Y, epsilon):
		grads = []

		for l_idx, l in enumerate(self.layers):
			layer_grad = np.zeros(l.W.shape)
			for i in range(l.W.shape[0]):
				for j in range(l.W.shape[1]):
					J_plus = self.get_loss_for_grad_check(X, Y, epsilon, l, l_idx, i, j)
					J_minus = self.get_loss_for_grad_check(X, Y, -epsilon, l, l_idx, i, j)
					J_check = (J_plus - J_minus)/(2*epsilon)
					layer_grad[i, j] = J_check
			grads.append(layer_grad)

		return grads