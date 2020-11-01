from ..components import SigmoidLayer

class LayerBuilder():

	@staticmethod
	def build_layer(layer_type, layer_size, input_weights):
		if layer_type == 'sigmoid':
			return SigmoidLayer(layer_size, input_weights)