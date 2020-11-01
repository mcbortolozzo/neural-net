

class Layer():

	def __init__(self, size, input_weights, bias_weights):
		self.size = size
		self.input_weights = input_weights
		self.bias_weights = bias_weights

class SigmoidLayer(Layer):
	
	def __init__(self, size, input_weights, bias_weights):
		super().__init__(size, input_weights, bias_weights)
