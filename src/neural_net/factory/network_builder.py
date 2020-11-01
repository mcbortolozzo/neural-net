import numpy as np

from ..components import NeuralNet
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
		bias_weights = []

		layer_weights_input = input_line.split(';')
		for neuron in layer_weights_input:
			neuron_weights = neuron.split(',')
			bias_weights.append(float(neuron_weights[0]))
			layer_weights.append([float(x) for x in neuron_weights[1:]])

		return layer_weights, bias_weights



	@staticmethod
	def parse_weights_file(layer_sizes, weights_file):
		input_weights = []
		bias_weights = []

		try:
			with open(weights_file, 'r') as f:
				lines = f.readlines()
				for size in layer_sizes[:-1]:
					current_line = lines.pop(0)
					layer_weights, layer_bias_weights = NetworkBuilder.parse_line_weights(size, current_line)
					input_weights.append(layer_weights)
					bias_weights.append(layer_bias_weights)

			return input_weights, bias_weights
		except Exception as e:
			print(e)
			raise ValueError("Invalid weights file")	


	@staticmethod
	def build_network_from_input_files(network_file, weights_file):
		lambd, layer_sizes = NetworkBuilder.parse_network_file(network_file)
		input_weights, bias_weights = NetworkBuilder.parse_weights_file(layer_sizes, weights_file)
		nnet = NeuralNet(lambd)
		for i in range(1, len(layer_sizes)):
			size = layer_sizes[i]
			layer_weights = (input_weights[i-1], bias_weights[i-1])
			layer = LayerBuilder.build_layer('sigmoid', size, layer_weights)
			nnet.add_layer(layer)

		return nnet
		

		

