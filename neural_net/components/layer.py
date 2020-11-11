import numpy as np

class Layer():

	def __init__(self, size, input_weights):
		self.size = size
		self.W = np.array(input_weights)
		self.in_mem = None
		self.z_mem = None
		self.a_mem = None
		self.d_mem = None
		self.grad_mem = None

	def activation(self, z):
		raise NotImplementedError()

	def get_squared_weights(self):
		#Exclude bias weights
		return self.W[:, 1:] ** 2

	def calculate_gradient(self, delta, lambd):
		m = delta.shape[0]
		P = lambd * np.hstack((np.zeros((self.W.shape[0], 1)), self.W[:, 1:]))
		return (P + delta.T @ self.in_mem)/m

	def backprop(self, delta, lambd, lr):
		self.d_mem = delta
		self.grad_mem = self.calculate_gradient(delta, lambd)

		dW = (delta @ self.W) * (self.in_mem * (1-self.in_mem))

		# Update weights
		self.W -= lr*self.grad_mem

		# Ignore bias component
		return dW[:, 1:]

	def add_bias(self, X):
		m = X.shape[0]
		return np.hstack((np.ones((m, 1)), X))

	def run_forward(self, X, W):
		X_bias = self.add_bias(X)
		Z = X_bias @ W.T
		A = self.activation(Z)
		return X_bias, Z, A

	def forward(self, X):
		self.in_mem, self.z_mem, self.a_mem = self.run_forward(X, self.W) 
		return self.a_mem


class SigmoidLayer(Layer):
	
	def __init__(self, size, input_weights):
		super().__init__(size, input_weights)

	def activation(self, z):
		return 1/(1 + np.exp(-z))

class LinearLayer(Layer):

	def __init__(self, size, input_weights):
		super().__init__(size, input_weights)

	def activation(self, z):
		return z
