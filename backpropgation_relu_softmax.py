import numpy as np

import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

  #Layer_initiallization

  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.01*np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))

  def forward(self, inputs):
    #remember input values
    self.inputs = inputs 
    #Calculate output values from inputs, weights and biases

    self.output = np.dot(inputs, self.weights) + self.biases

  def backward(self, dvalues):
    self.dweights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis =0, keepdims = True)
    # gradient on vlaues 

    self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
  
  #forward pass
  def forward(self, inputs):
    self.inputs = inputs

    #calculate output values from inputs
    self.output = np.maximum(0,inputs)

  def backward(self, dvalues):
    #modify original variable
    self.dinputs = dvalues.copy()

    #zero gradient where input values were negative
    self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
  #forward pass

  def forward(self, inputs):
    #remember the inputs 
    self.inputs = inputs

    #get unnormalized probab
    exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims= True))

    #normalize them for each sample
    probablities = exp_values/np.sum(exp_values, axis = 1, keepdims= True)

    self.output = probablities

#backword pass
  def backward(self, dvalues):
    self.inputs = np.empty_like(dvalues)

    for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
      single_output = single_output.reshape(-1,1)
      jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

      self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
  def calculate(self, output, y):
    sample_losses = self.forward(output, y)

    data_loss = np.mean(sample_losses)

    return data_loss

#Cross-entropy loss

class Loss_CategoricalCrossentropy(Loss):
  #forward pass

  def forward(self, y_pred, y_true):
    #number of samples in a batch
    samples = len(y_pred)
    # clip data to prevent division by 0 clop both sides to not drag mean towards any values
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    #prob for target values only if categorical labels
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples),y_true]
    elif len(y_true.shape) == 2:
      corrent_confidences = np.sum(y_pred_clipped * y_true, axis = 1)
    
    #Losses 

    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods

#Backward pass

  def backward(self, dvalues, y_true):
    samples = len(dvalues)

    #numbers of labels in every samples 

    labels = len(dvalues[0])

    if len(y_true.shape) == 1:
      y_true = np.eye(labels)[y_true]


    self.dinputs = -y_true / dvalues 

    #Normalize gradient 

    self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CaterogricalCrossentropy():

  #creates acitvation and loss function objects 

  def __init__(self):
    self.activation = Activation_Softmax()

    self.loss = Loss_CategoricalCrossentropy()

  def forward(self, inputs, y_true):
    #output layer's activation function
    self.activation.forward(inputs)

    #set the output

    self.output = self.activation.output

    #calcualate and return loss values

    return self.loss.calculate(self.output, y_true)

  #define backward pass
  def backward(self, dvalues, y_true):

    #number of samples
    samples = len(dvalues)

    #if labels are one-hot encoded, turn them into discrete values
    if len(y_true.shape) ==2 :
      y_true = np.argmax(y_true, axis = 1)

    #copy so we can safetly modify

    self.dinputs = dvalues.copy()

    #calcuate gradient

    self.dinputs[range(samples), y_true] -= 1

    #normalize gradient
    self.dinputs = self.dinputs / samples

#create dataset
X, y = spiral_data(samples = 100, classes = 3)

#create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

#create ReLU activation 
activation1 = Activation_ReLU()

#create second dense layer with 3 input feature and 3 output values
dense2 = Layer_Dense(3,3)

#create Softmax classifer's combined loss and activation
loss_activation = Activation_Softmax_Loss_CaterogricalCrossentropy()

#perform a forward pass of our trianing data through this layer
dense1.forward(X)

#perform a forward pass through activation function take s the output of first dnese layer here
activation1.forward(dense1.output)

#perform a forward pass through activation 
dense2.forward(activation1.output)

loss = loss_activation.forward(dense2.output, y)

print(loss_activation.output[:5])

#print loss values
print('loss', loss)

predictions = np.argmax(loss_activation.output, axis =1)
if len(y.shape) == 2:
  y = np.argmax(y, axis = 1)
accuracy = np.mean(predictions ==y)

#print accuracy 
print('acc : ', accuracy)

#backward pass

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)


#Print gradients

print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)


