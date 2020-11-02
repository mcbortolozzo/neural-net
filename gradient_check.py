import click

from neural_net.factory import NetworkBuilder, DatasetBuilder

EPSILON = 0.000001

@click.command()
@click.argument('network_file')
@click.argument('weight_file')
@click.argument('dataset_file')
def gradient_checking(network_file, weight_file, dataset_file):
	nnet = NetworkBuilder.build_network_from_input_files(network_file, weight_file)
	X, Y = DatasetBuilder.build_dataset_from_input_file(dataset_file)

	gradients = nnet.verify_gradient(X, Y, EPSILON)	

	layers = []
	for layer in gradients:
		neurons = []
		for i in range(layer.shape[0]):
			neurons.append(', '.join(["%.5f" % x for x in layer[i, :]]))
	
		layers.append('; '.join(neurons))

	output = '\n'.join(layers)	
	print(output)		

if __name__ == '__main__':
	gradient_checking()