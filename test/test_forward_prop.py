from neural_net import NetworkBuilder, ClassificationNeuralNet

from test.const import *

import numpy as np
import unittest


class TestForwardPropagation(unittest.TestCase):

	def test_forward_activation_small_network_first_input(self):
		x1 = np.array([[0.13]])

		expected_z2 = np.array([0.413, 0.326])
		expected_a2 = np.array([0.60181, 0.58079])

		expected_z3 = np.array([1.34937])
		expected_out = np.array([0.79403])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		output = nnet.forward(x1)

		self.assertTrue(np.isclose(nnet.layers[0].z_mem, expected_z2).all())
		self.assertTrue(np.isclose(nnet.layers[0].a_mem, expected_a2).all())

		self.assertTrue(np.isclose(nnet.layers[1].z_mem, expected_z3).all())
		self.assertTrue(np.isclose(output, expected_out).all())


	def test_forward_activation_small_network_first_input(self):
		x1 = np.array([[0.42]])

		expected_z2 = np.array([0.442, 0.384])
		expected_a2 = np.array([0.60874, 0.59484])

		expected_z3 = np.array([1.36127])
		expected_out = np.array([0.79597])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		output = nnet.forward(x1)

		self.assertTrue(np.isclose(nnet.layers[0].z_mem, expected_z2).all())
		self.assertTrue(np.isclose(nnet.layers[0].a_mem, expected_a2).all())

		self.assertTrue(np.isclose(nnet.layers[1].z_mem, expected_z3).all())
		self.assertTrue(np.isclose(output, expected_out).all())


	def test_forward_activation_large_network_first_input(self):
		x1 = np.array([[0.32, 0.68]])

		expected_z2 = np.array([0.74, 1.1192, 0.35640, 0.87440])
		expected_a2 = np.array([0.677, 0.75384, 0.58817, 0.70566])

		expected_z3 = np.array([1.94769, 2.12136, 1.48154])
		expected_a3 = np.array([0.87519, 0.89296, 0.81480])

		expected_z4 = np.array([1.60831, 1.66805])
		expected_out = np.array([0.83318, 0.84132])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)
		output = nnet.forward(x1)

		self.assertTrue(np.isclose(nnet.layers[0].z_mem, expected_z2).all())
		self.assertTrue(np.isclose(nnet.layers[0].a_mem, expected_a2).all())

		self.assertTrue(np.isclose(nnet.layers[1].z_mem, expected_z3).all())
		self.assertTrue(np.isclose(nnet.layers[1].a_mem, expected_a3).all())

		self.assertTrue(np.isclose(nnet.layers[2].z_mem, expected_z4).all())
		self.assertTrue(np.isclose(output, expected_out).all())

	def test_forward_activation_large_network_second_input(self):
		x1 = np.array([[0.83, 0.02]])

		expected_z2 = np.array([0.5525, 0.81380, 0.17610, 0.60410])
		expected_a2 = np.array([0.63472, 0.69292, 0.54391, 0.64659])

		expected_z3 = np.array([1.81696, 2.02468, 1.37327])
		expected_a3 = np.array([0.86020, 0.88336, 0.79791])

		expected_z4 = np.array([1.58228, 1.64577])
		expected_out = np.array([0.82953, 0.83832])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)
		output = nnet.forward(x1)

		self.assertTrue(np.isclose(nnet.layers[0].z_mem, expected_z2).all())
		self.assertTrue(np.isclose(nnet.layers[0].a_mem, expected_a2).all())

		self.assertTrue(np.isclose(nnet.layers[1].z_mem, expected_z3).all())
		self.assertTrue(np.isclose(nnet.layers[1].a_mem, expected_a3).all())

		self.assertTrue(np.isclose(nnet.layers[2].z_mem, expected_z4).all())
		self.assertTrue(np.isclose(output, expected_out).all())

	def test_cost_function_calculation_first_example_no_regularization(self):
		network_output = np.array([[0.79403], [0.79597]])
		y = np.array([[0.9], [0.23]])

		expected_loss = 0.82098

		nnet = ClassificationNeuralNet(0)
		actual_loss = nnet.loss(network_output, y)

		self.assertLess(np.abs(actual_loss-expected_loss),  TOLERANCE)

	def test_cost_function_calculation_second_example_no_regularization(self):
		network_output = np.array([[0.83318, 0.84132], [0.82953, 0.83832]])
		y = np.array([[0.75, 0.98], [0.75, 0.28]])

		expected_loss = 1.3675

		nnet = ClassificationNeuralNet(0)
		actual_loss = nnet.loss(network_output, y)

		self.assertLess(np.abs(actual_loss-expected_loss),  TOLERANCE)

	def test_cost_function_calculation_second_example_with_regularization(self):
		network_output = np.array([[0.83318, 0.84132], [0.82953, 0.83832]])
		y = np.array([[0.75, 0.98], [0.75, 0.28]])

		expected_loss = 1.90351

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)
		actual_loss = nnet.loss(network_output, y)

		self.assertLess(np.abs(actual_loss-expected_loss), TOLERANCE)


if __name__ == 'main':
	unittest.main()