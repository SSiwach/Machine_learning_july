#cross entory loss

class Loss_CategoricalCrossentropy(Loss):
  #Backward pass

  samples = len(dvalues)

  labels = len(dvalues[0])

  if len(y_true.shape) == 1:
    y_true = np.eye(labels)[y_true]

  #calculate gradient
  self.dinputs = -y_true / dvalues
  #Normalize gradient
  self.dinputs = self.dinputs/samples
