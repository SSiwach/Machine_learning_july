#ReLU activation

class Activation_ReLU:

  #forward pass
  def forward(self, inputs):
    #Remember input values

    self.inputs = inputs
    self.outputs = np.maximum(0,inputs)

  #backwork pass
  def backword(self, dvalues):
    self.dinputs = dvalues.copy()
    # make a copy of original variables, copy of the values first

    self.dinputs[self.inputs <= 0]  = 0
    
