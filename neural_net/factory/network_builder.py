import numpy as np

from ..components import ClassificationNeuralNet, RegressionNeuralNet
from .layer_builder import LayerBuilder

class NetworkBuilder():

	@staticmethod
	def parse_network_file(network_file):
		lambd = None
		layer_sizes = []

		try:
			with open(network_file, 'r') as f:
				lines = f.readlines()
				lambd = float(lines[0])
				for line in lines[1:]:
					layer_sizes.append(int(line))

			return lambd, layer_sizes
		except Exception as e:
			print(e)
			raise ValueError("Invalid network file")	

	@staticmethod
	def parse_line_weights(layer_size, input_line):
		layer_weights = []

		layer_weights_input = input_line.split(';')
		for neuron in layer_weights_input:
			neuron_weights = neuron.split(',')
			layer_weights.append([float(x) for x in neuron_weights])

		return layer_weights



	@staticmethod
	def parse_weights_file(layer_sizes, weights_file):
		input_weights = []

		try:
			with open(weights_file, 'r') as f:
				lines = f.readlines()
				for size in layer_sizes[:-1]:
					current_line = lines.pop(0)
					input_weights.append(NetworkBuilder.parse_line_weights(size, current_line))

			return input_weights
		except Exception as e:
			print(e)
			raise ValueError("Invalid weights file")	


	@staticmethod
	def build_network_from_input_files(network_file, weights_file):
		lambd, layer_sizes = NetworkBuilder.parse_network_file(network_file)
		input_weights = NetworkBuilder.parse_weights_file(layer_sizes, weights_file)
		nnet = ClassificationNeuralNet(lambd)
		for i in range(0, len(layer_sizes) - 1):
			size = layer_sizes[i]
			layer_weights = (input_weights[i])
			layer = LayerBuilder.build_layer('sigmoid', size, layer_weights)
			nnet.add_layer(layer)

		return nnet

	@staticmethod
	def build_network_from_specs(lambd, lr, layer_specs, regression=False):
		if regression:
			nnet = RegressionNeuralNet(lambd, lr)
		else:
			nnet = ClassificationNeuralNet(lambd, lr)
		prev_layer_size = layer_specs[0]['size']
		for layer in layer_specs[1:]:
			layer_shape = (layer['size'], prev_layer_size + 1)
			l = LayerBuilder.initialize_random_layer(layer['type'], layer_shape)
			nnet.add_layer(l)
			prev_layer_size = layer['size']

		return nnet
		

		

