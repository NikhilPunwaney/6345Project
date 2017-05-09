import scipy.io
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy_loss(a,b):
	loss = np.dot(a, np.log(b))
	return -loss

mnist = scipy.io.loadmat('mnist_small.mat')


N_train_dim = mnist['train_X'].shape
N_test_dim = mnist['test_X'].shape
N_classes = mnist['train_Y'].shape

train_bias = np.ones((10000,1))
test_bias = np.ones((1000,1))

train_X = np.append(mnist['train_X'], train_bias, axis=1)
test_X = np.append((mnist['test_X'])/255.0, test_bias, axis=1)
train_Y = mnist['train_Y']
test_Y = mnist['test_Y']



N_hidden = 128

Wh = np.random.rand(N_hidden, train_X.shape[1]) / train_X.shape[1]
Wo = np.random.rand(10, N_hidden+1) / (N_hidden+1)

#Training Loop

N_iters = 100000
step_size = 0.001
momentum = 0.9
train_loss = np.zeros((N_iters, 1))
dropout_rate = 0.8

# Declare some variables to hold the previous gradient (for momentum)
v_dL_dWo = np.zeros((Wo.shape)) 
v_dL_dWh = np.zeros((Wh.shape)) 

for i in range(N_iters):
	print "training", i
#  ----- 6.345 TODO: Fill this in!
#    Here's some comments to guide you...
#    % The number of spaces between the comments indicate _approximately_
#    % how many lines each step should require
#    % Randomly sample a single training pair (x, y)
	rand = np.random.randint(0, 10000)
	x = train_X[rand]
	#print x.shape
	y = train_Y[rand]
	#print y

	# Forward pass the batch
	# Don't forget to append the implicit bias to the hidden vector	
	
	neuron_matrix = np.matmul(Wh, x)
	#print neuron_matrix
	relu_matrix = neuron_matrix * (neuron_matrix > 0)

	relu_gradients = np.ones(len(neuron_matrix)) * (neuron_matrix > 0)

	# print relu_matrix.shape

	droput_number = int(N_hidden * (1 - dropout_rate))
	
	dropout_array = np.random.choice(N_hidden, droput_number, replace=False)
	
	if droput_number>0:
		relu_matrix[dropout_array] = 0.0
		
	

	
	relu_matrix_with_bias = np.append(relu_matrix, 1)

	


	#print relu_matrix_with_bias.shape

	output = np.matmul(Wo, relu_matrix_with_bias)

	#print output.shape


	#Compute the softmax outputs (taking into account numerical stability)

	softmax_logits = softmax(output)

	#print softmax_logits

	#Compute loss L, store it in our train_loss buffer for this iter

	L = cross_entropy_loss(y, softmax_logits)
	train_loss[i] = L
	print L

	#Now backprop gradients to find dL_dWo and dL_dWh

	dL_dWo = np.matmul(np.reshape((softmax_logits - y), [10,1]), np.reshape(relu_matrix_with_bias, [1,129]))
	dL_dH = np.matmul((softmax_logits - y), Wo)

	dL_dH = dL_dH[0:128]


	dL_dHb = np.multiply(dL_dH, relu_gradients)

	
	# print relu_gradients.shape

	# print dL_dWo.shape
	# print dL_dH.shape


	
	dL_dWh = np.matmul(np.reshape(dL_dHb, [128,1]), np.reshape(x, [1,785]))



	#Take a gradient step to update Wo and Wh

	v_dL_dWo = step_size * dL_dWo + momentum * v_dL_dWo
	v_dL_dWh = step_size * dL_dWh + momentum * v_dL_dWh
	Wo = Wo - v_dL_dWo
	Wh = Wh - v_dL_dWh

	
#Now compute the testing accuracy
# print "Wo", Wo
# print "Wh", Wh

num_correct_test = 0

for i in range(test_X.shape[0]):
	
	print "testing", i
	x = test_X[i]
	y = test_Y[i]

	neuron_matrix = np.matmul(Wh, x)
	relu_matrix = neuron_matrix * (neuron_matrix > 0)
	relu_matrix *= dropout_rate

	relu_matrix_with_bias = np.append(relu_matrix, 1)
	
	output = np.matmul(Wo, relu_matrix_with_bias)

	softmax_logits = softmax(output)

	prediction = np.argmax(softmax_logits)

	print "prediction", prediction

	truth = np.argmax(y)

	print "truth", truth

	if prediction == truth:
		num_correct_test +=1

print "test accuracy", (1.0 * num_correct_test) / test_X.shape[0]



	






