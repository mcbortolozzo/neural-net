

class NeuralNet():

	def __init__(self, lambd):
		self.lambd = lambd
		self.layers = []

	def add_layer(self, layer):
		self.layers.append(layer)

	def forward(self, input_activation):
		activation = input_activation
		for l in self.layers:
			activation = l.forward(activation)

		return activation