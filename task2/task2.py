import numpy as np


class Layer:

	def __init__(self, input_size, output_size):
		self.node_values = [1 for _ in range(input_size)]
		self.weights = np.random.normal(0, 1.0, size=(input_size, output_size))
		self.next_layer = None
		self.prev_layer = None

	def set_next_layer(self, layer):
		self.next_layer = layer;

	def forward(self):
		output_nodes = np.matmul(self.node_values, self.weights)
		self.transition_func(output_nodes)
		if self.next_layer == None:
			return output_nodes
		self.next_layer.node_values = output_nodes
		return self.next_layer.forward()

	def transition_func(self, vals):
		for i in range(len(vals)):
			vals[i] = 0 if vals[i] < 0 else vals[i]
		

	def update_weights(self):
		pass
	

class NeuralNetwork:

	def __init__(self, layer_sizes):
		self.layers = []
		for i in range(len(layer_sizes)-1):
			new_layer = Layer(layer_sizes[i], layer_sizes[i+1])
			self.layers.append(new_layer)
		
		# Set up layer connections
		for i in range(len(self.layers) - 1):
			self.layers[i].next_layer = self.layers[i+1]
		for i in range(len(self.layers)-1, -1, -1):
			self.layers[i].prev_layer = self.layers[i-1]

	def forward(self):
		return self.layers[0].forward()

		
		
			

nn = NeuralNetwork([2,3,4, 100])
print(nn.forward())
#print(nn.layers[0].forward())