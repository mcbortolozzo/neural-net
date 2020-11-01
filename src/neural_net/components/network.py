

class NeuralNet():

	def __init__(self, lambd):
		self.lambd = lambd
		self.layers = []

	def add_layer(self, layer):
		self.layers.append(layer)