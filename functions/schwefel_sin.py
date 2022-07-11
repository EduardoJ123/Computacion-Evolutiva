import numpy as np

def schwefel_sin(x, y):
	resultado = 0
	X = []
	X.append(x)
	X.append(y)
	for idx in range(len(X)):
		resultado = resultado + (X[idx]*np.sin(np.sqrt(np.abs(X[idx]))))
	return (-1 * resultado)