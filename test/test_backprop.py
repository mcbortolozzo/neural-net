from neural_net import NetworkBuilder

from test.const import *

import unittest
import numpy as np

class TestBackprop(unittest.TestCase):

	def test_backprop_small_network_first_input(self):
		X = np.array([[0.13]])
		Y = np.array([[0.9]])

		expected_delta_3 = np.array([[-0.10597]])
		expected_delta_2 = np.array([[-0.01270, -0.01548]])

		expected_gradients_theta_2 = np.array([[-0.1057, -0.06378, -0.06155]])
		expected_gradients_theta_1 = np.array([[-0.01270, -0.00165], [-0.01548, -0.00201]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		nnet.forward(X)
		nnet.backprop(Y)

		self.assertTrue(np.isclose(nnet.layers[-1].d_mem, expected_delta_3, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-1].grad_mem, expected_gradients_theta_2, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-2].d_mem, expected_delta_2, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-2].grad_mem, expected_gradients_theta_1, atol=TOLERANCE).all())

	def test_backprop_small_network_second_input(self):
		X = np.array([[0.42]])
		Y = np.array([[0.23]])

		expected_delta_3 = np.array([[0.56597]])
		expected_delta_2 = np.array([[0.06740, 0.08184]])

		expected_gradients_theta_2 = np.array([[0.56597, 0.34452, 0.33666]])
		expected_gradients_theta_1 = np.array([[0.06740, 0.02831], [0.08184, 0.03437]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		nnet.forward(X)
		nnet.backprop(Y)

		self.assertTrue(np.isclose(nnet.layers[-1].d_mem, expected_delta_3, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-1].grad_mem, expected_gradients_theta_2, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-2].d_mem, expected_delta_2, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-2].grad_mem, expected_gradients_theta_1, atol=TOLERANCE).all())

	def test_backprop_large_network_both_inputs(self):
		X = np.array([[0.32, 0.68], [0.83, 0.02]])
		Y = np.array([[0.75, 0.98], [0.75, 0.28]])

		expected_gradients_theta_1 = np.array([[0.00804, 0.02564, 0.04987], [0.00666, 0.01837, 0.06719], [0.00973, 0.03196, 0.05252], [0.00776, 0.05037, 0.08492]])
		expected_gradients_theta_2 = np.array([[0.01071, 0.09068, 0.02512, 0.12597, 0.11586], [0.02442, 0.06780, 0.04164, 0.05308, 0.12677], [0.03056, 0.08924, 0.12094, 0.10270, 0.03078]])
		expected_gradients_theta_3 = np.array([[0.08135, 0.17935, 0.12476, 0.13186], [0.20982, 0.19195, 0.30434, 0.25249]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)
		nnet.forward(X)
		nnet.backprop(Y)

		self.assertTrue(np.isclose(nnet.layers[-1].grad_mem, expected_gradients_theta_3, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-2].grad_mem, expected_gradients_theta_2, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(nnet.layers[-3].grad_mem, expected_gradients_theta_1, atol=TOLERANCE).all())

	def test_numerical_gradient_validation_small_network(self):
		epsilon = 0.000001
		X = np.array([[0.13], [0.42]])
		Y = np.array([[0.9], [0.23]])

		expected_gradient_theta_1 = np.array([[0.02735, 0.01333], [0.03318, 0.01618]])
		expected_gradient_theta_2 = np.array([[0.23, 0.14037, 0.13756]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		actual_grads = nnet.verify_gradient(X, Y, epsilon)

		self.assertTrue(np.isclose(actual_grads[0], expected_gradient_theta_1, atol=TOLERANCE).all())
		self.assertTrue(np.isclose(actual_grads[1], expected_gradient_theta_2, atol=TOLERANCE).all())

	def test_backprop_compared_to_gradient_checking(self):
		epsilon = 0.000001
		X = np.array([[0.13], [0.42]])
		Y = np.array([[0.9], [0.23]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_1, WEIGHTS_DEFINITION_TEST_FILE_1)
		grad_check = nnet.verify_gradient(X, Y, epsilon)

		nnet.forward(X)
		nnet.backprop(Y)

		self.assertLess(np.mean(nnet.layers[0].grad_mem - grad_check[0]), 10e-8)
		self.assertLess(np.mean(nnet.layers[1].grad_mem - grad_check[1]), 10e-8)

	def test_backprop_compared_to_gradient_checking_large(self):
		epsilon = 0.000001
		X = np.array([[0.32, 0.68], [0.83, 0.02]])
		Y = np.array([[0.75, 0.98], [0.75, 0.28]])

		nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE_2, WEIGHTS_DEFINITION_TEST_FILE_2)
		grad_check = nnet.verify_gradient(X, Y, epsilon)

		nnet.forward(X)
		nnet.backprop(Y)

		self.assertLess(np.mean(nnet.layers[0].grad_mem - grad_check[0]), 10e-8)
		self.assertLess(np.mean(nnet.layers[1].grad_mem - grad_check[1]), 10e-8)


