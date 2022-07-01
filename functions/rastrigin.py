import numpy as np

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)