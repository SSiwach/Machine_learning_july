class Layer_Dense:

  def __init__(self, inputs, neurons):
    self.weights = 0.01 * np.random.rand(inputs, neurons)
    self.biases = np.zeros(1,neurons)
  #forward pass
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases
#ReLU activation
class Activation_ReLU:

  #Forward pass
  def forward(self, inputs):
    self.outputs = np.maximum(0, inputs)

  def backward(self, dvalues):
    self.weights = np.dot(self.inputs.T, dvalues)
    self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
    #gradient on values 

    self.dinputs = np.dot(dvalues, self.weights.T)
