from neural_net import NetworkBuilder

NETWORK_DEFINITION_TEST_FILE = './data/test/network_definition.txt'
WEIGHTS_DEFINITION_TEST_FILE = './data/test/weights_definition.txt'

def test_layer_size_parser():
	expected_lambda = 0.1
	expected_sizes = [1, 2, 1]

	actual_lambda, actual_sizes = NetworkBuilder.parse_network_file(NETWORK_DEFINITION_TEST_FILE)

	assert actual_lambda == expected_lambda
	assert actual_sizes == expected_sizes

def test_layer_weights_parser():
	expected_input_weights = [[[0.1], [0.2]], [[0.5, 0.6]]]
	expected_bias_weights = [[0.4, 0.3], [0.7]]
	layer_sizes = [1, 2, 1]

	actual_input_weights, actual_bias_weights = NetworkBuilder.parse_weights_file(layer_sizes, WEIGHTS_DEFINITION_TEST_FILE)

	assert actual_input_weights == expected_input_weights
	assert actual_bias_weights == expected_bias_weights

def test_build_network_from_input_files():
	expected_size_first_layer = 2
	expected_input_weights_first_layer = [[0.1], [0.2]]
	expected_bias_weights_first_layer = [0.4, 0.3]

	expected_size_second_layer = 1
	expected_input_weights_second_layer = [[0.5, 0.6]]
	expected_bias_weights_second_layer = [0.7]

	nnet = NetworkBuilder.build_network_from_input_files(NETWORK_DEFINITION_TEST_FILE, WEIGHTS_DEFINITION_TEST_FILE)

	assert len(nnet.layers) == 2
	assert nnet.layers[0].size == expected_size_first_layer
	assert nnet.layers[0].input_weights == expected_input_weights_first_layer
	assert nnet.layers[0].bias_weights == expected_bias_weights_first_layer
	assert nnet.layers[1].size == expected_size_second_layer
	assert nnet.layers[1].input_weights == expected_input_weights_second_layer
	assert nnet.layers[1].bias_weights == expected_bias_weights_second_layer


test_layer_size_parser()
test_layer_weights_parser()
test_build_network_from_input_files()