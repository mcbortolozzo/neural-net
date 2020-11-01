from ..components import SigmoidLayer

class LayerBuilder():

	@staticmethod
	def build_sigmoid_layer(layer_size, layer_weights):
		input_weights, bias_weights = layer_weights

		return SigmoidLayer(layer_size, input_weights, bias_weights)


	@staticmethod
	def build_layer(layer_type, layer_size, layer_weights):
		if layer_type == 'sigmoid':
			return LayerBuilder.build_sigmoid_layer(layer_size, layer_weights)
