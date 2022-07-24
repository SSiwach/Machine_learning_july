import numpy as np

np.eye(5)

np.eye(5)[1]

softmax_output = [0.7, 0.1, 0.2]

softmax_output = np.array(softmax_output).reshape(-1,1)

print(softmax_output)

print(np.eye(softmax_output.shape[0]))

print(softmax_output*np.eye(softmax_output.shape[0]))


print(np.diagflat(softmax_output))


print(np.dot(softmax_output, softmax_output.T))
