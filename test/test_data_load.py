from neural_net import NetworkBuilder

from test.const import *

import numpy as np
import unittest


class TestDataParser(unittest.TestCase):

	def test_layer_size_parser(self):
		expected_lambda = 0
		expected_sizes = [1, 2, 1]

		actual_lambda, actual_sizes = NetworkBuilder.parse_network_file(NETWORK_DEFINITION_TEST_FILE_1)

		self.assertEqual(actual_lambda, expected_lambda)
		self.assertEqual(actual_sizes, expected_sizes)

	def test_layer_weights_parser(self):
		expected_input_weights = [[[0.4, 0.1], [0.3, 0.2]], [[0.7, 0.5, 0.6]]]
		layer_sizes = [1, 2, 1]

		actual_input_weights = NetworkBuilder.parse_weights_file(layer_sizes, WEIGHTS_DEFINITION_TEST_FILE_1)

		self.assertEqual(actual_input_weights, expected_input_weights)

	def test_build_network_from_input_files(self):
		expected_size_first_layer = 1
		expected_input_weights_first_layer = np.array([[0.4, 0.1], [0.3, 0.2]])

		expected_size_second_layer = 2
		expected_input_weights_second_layer = np.array([[0.7, 0.5, 0.6]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)

		self.assertEqual(len(nnet.layers), 2)
		self.assertEqual(nnet.layers[0].size, expected_size_first_layer)
		self.assertTrue((nnet.layers[0].W == expected_input_weights_first_layer).all())
		self.assertEqual(nnet.layers[1].size, expected_size_second_layer)
		self.assertTrue((nnet.layers[1].W == expected_input_weights_second_layer).all())

	def test_build_network_from_input_files_large_network(self):
		expected_size_first_layer = 2
		expected_input_weights_first_layer = np.array([[0.42, 0.15, 0.4], [0.72, 0.1, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])

		expected_size_second_layer = 4
		expected_input_weights_second_layer = np.array([[0.21, 0.67, 0.14, 0.96, 0.87], [0.87, 0.42, 0.2, 0.32, 0.89], [0.03, 0.56, 0.8, 0.69, 0.09]])

		expected_size_third_layer = 3
		expected_input_weights_third_layer = np.array([[0.04, 0.87, 0.42, 0.53], [0.17, 0.1, 0.95, 0.69]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)

		self.assertEqual(len(nnet.layers), 3)
		self.assertEqual(nnet.layers[0].size, expected_size_first_layer)
		self.assertTrue((nnet.layers[0].W == expected_input_weights_first_layer).all())
		self.assertEqual(nnet.layers[1].size, expected_size_second_layer)
		self.assertTrue((nnet.layers[1].W == expected_input_weights_second_layer).all())
		self.assertEqual(nnet.layers[2].size, expected_size_third_layer)
		self.assertTrue((nnet.layers[2].W == expected_input_weights_third_layer).all())

if __name__ == 'main':
	unittest.main()