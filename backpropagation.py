import click

from neural_net.factory import NetworkBuilder, DatasetBuilder

@click.command()
@click.argument('network_file')
@click.argument('weight_file')
@click.argument('dataset_file')
def backpropagation(network_file, weight_file, dataset_file):
	nnet = NetworkBuilder.build_network_from_input_files(network_file, weight_file)
	X, Y = DatasetBuilder.build_dataset_from_input_file(dataset_file)

	nnet.forward(X)
	nnet.backprop(Y)

	layers = []
	for layer in nnet.layers:
		neurons = []
		for i in range(layer.grad_mem.shape[0]):
			neurons.append(', '.join(["%.5f" % x for x in layer.grad_mem[i, :]]))
	
		layers.append('; '.join(neurons))

	output = '\n'.join(layers)	
	print(output)		

if __name__ == '__main__':
	backpropagation()