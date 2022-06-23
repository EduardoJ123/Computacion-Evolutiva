import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import randint
from numpy import linspace
from mpl_toolkits import mplot3d

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)

def sphere(x, y):
	return (x + 2)**2 + (y + 2)**2

def ackley(x, y):
	return (-20)*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + 20 + np.exp(1)

def PSO(xu, xl, f):
	w = 0.6
	c1 = 1.5
	c2 = 2.5
	X = []		#Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	V = []		#Vector de velocidades
	#Inicializar N individuos aleatoriamente y un vector semejante de velocidades
	for i in range(N):
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
		V.append((random.uniform(xl,xu), random.uniform(xl,xu)))
	Xb = X
	for gen in range(g):
		for i in range(N):
			if call_f(X[i][0], X[i][1], f) < call_f(Xb[i][0], Xb[i][1], f):
				Xb[i] = X[i]
		xg = find_best(Xb, f)
		for i in range(N):
			r1 = random.uniform(0,1)
			r2 = random.uniform(0,1)
			op1 = (w*V[i][0], w*V[i][1])				#Primer término
			op2 = (Xb[i][0]-X[i][0], Xb[i][1]-X[i][1])	#Segundo término (1/2)
			op3 = (r1*c1*op2[0], r1*c1*op2[1])			#Segundo término (2/2)
			op4 = (xg[0]-X[i][0], xg[1]-X[i][1])		#Tercer término (1/2)
			op5 = (r2*c2*op4[0], r2*c2*op4[1])			#Tercer término (2/2)
			V[i] = (op1[0]+op3[0]+op5[0], op1[1]+op3[1]+op5[1])
			X[i] = (X[i][0]+V[i][0], X[i][1]+V[i][1])
	#Encontrar y retornar la mejor solución del conjunto final
	minimo = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(N):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			minimo = X[idx]
			valor = valor_t
	return minimo, valor

#Función que retorna la tupla de la forma (x, y)
#correspondiente a la mejor solución para una
#población y función dadas
def find_best(X, f):
	best = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(len(X)):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			best = X[idx]
			valor = valor_t
	return best

def call_f(arg1, arg2, sel):
	if sel == 0:
		return sphere(arg1, arg2)
	elif sel == 1:
		return rastrigin(arg1, arg2)
	elif sel == 2:
		return ackley(arg1, arg2)
	else:
		return 0

N = 200
g = 500
rep = 30
iteraciones = []
#Sphere
resultados_z = []
resultados_x = []
resultados_y = []
for idx in range(rep):
	res, val = PSO(-5, 5, 0)
	print("Sphere (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	iteraciones.append(idx+1)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar sphere
fig = plt.figure()
fig.suptitle('Sphere', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sphere(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()

#Rastrigin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = PSO(-5, 5, 1)
	print("Rastrigin (rep:", idx, "): ", res, " = ", val)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar rastrigin
fig = plt.figure()
fig.suptitle('Rastrigin', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()

#Ackley
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = PSO(-5, 5, 2)
	print("Ackley (rep:", idx, "): ", res, " = ", val)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Ackley
fig = plt.figure()
fig.suptitle('Ackley', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(ackley(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()