import tensorflow as tf 
'''
input-> weight-> hidden layer 1(activation function)-> weights-> hidden layer 2(activation function)->weights->output layer

Its a feed forward neural network

compare output to intended output-> cost function (cross entropy)

optimization function (optimizer)->minimize cost(AdamOptimizer,SGD,AdaGrad)

backpropagation

feed forward+backpropagation=epoch

'''

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)
'''
one_hot encoding-
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
'''
#define the number of nodes in hidden layer 1
n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500
 
#define the number of classes
n_classes=10
#define the batch size since the data size is too large, and hence we need to take data into small batches
batch_size=100
# height * width
#height is 0 and we have 28*28 pixel images, so width=28*28=784
x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

def neural_network_model(data):
	#generate random weights for hidden_1_layer
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl1))}

	#we use biases so that we can handle zero values inputs
	#input*weights + biases

	hidden_2_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl2))}
	hidden_3_layer={'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
	'biases':tf.Variable(tf.random_normal(n_nodes_hl3))}
	output_layer={'weights':tf.Variable(tf.random_normal([784,n_nodes_hl1])),
	'biases':tf.Variable(tf.random_normal([n_classes]))}
	"""
	for the first layer, input is the raw data
	for the second layer, output from the first layer is feed as input to the second layer and so on 
	"""
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']+hidden_1_layer['biases']))
	#apply activation function
	l1=tf.nn.relu(l1)

	l2=tf.add(tf.matmul(l1,hidden_2_layer['weights']+hidden_2_layer['biases']))
	#apply activation function
	l2=tf.nn.relu(l2)

	l3=tf.add(tf.matmul(l2,hidden_3_layer['weights']+hidden_3_layer['biases']))
	#apply activation function
	l3=tf.nn.relu(l3)

	output=tf.add(tf.matmul(l3,output_layer['weights']+output_layer['biases']))
	return output





