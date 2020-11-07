import numpy as np

from ..components import SigmoidLayer

class LayerBuilder():

	@staticmethod
	def build_layer(layer_type, layer_size, input_weights):
		if layer_type == 'sigmoid':
			return SigmoidLayer(layer_size, input_weights)

	@staticmethod
	def initialize_random_layer(layer_type, layer_shape):
		input_weights = 2*np.random.rand(*layer_shape) - 1
		return LayerBuilder.build_layer(layer_type, layer_shape[0], input_weights)