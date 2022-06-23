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

def schwefel(x, y):
	resultado = 0
	X = []
	X.append(x)
	X.append(y)
	for idx in range(len(X)):
		resultado = resultado + (X[idx]*np.sin(np.sqrt(np.abs(X[idx]))))
	return (418.9829*2) - resultado

def BFO(xl, xu, f):
	c = 0.1 	#Tamaño de paso
	N = 50		#Tamaño de la población
	Nc = 100	#Número de pasos de quimiotaxis
	Ns = 4 		#Número pasos de nado (cost-reduction)
	Nr = 4 		#Número de pasos de reproducción
	Ne = 2 		#Número de pasos de eliminación/dispersión
	d = 1 		#Profundidad fuerza de atracción
	h = 1 		#Profundidad fuerza de repulsión
	wa = 0.2 	#Anchura fuerza de atracción
	wr = 10 	#Anchura fuerza de repulsión
	pe = 0.25	#Probabilidad de eliminación/dispersión
	X = []		##Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	Fi = []		##Lista de posibles soluciones más su valor
	delta = [0, 0]	#Vector n-dimensional aleatorio en el rango -1:1
	#Inicializar población N random 
	for i in range(N): 
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
		Fi.append([X[i][0], X[i][1], 0])
	for l in range(Ne): #Ciclo de eliminación/dispersión
		for k in range(Nr): #Ciclo de reproducción
			for j in range(Nc): #Ciclo de quimiotaxis
				for idx in range(N): #Ciclo por cada elemento de la población
					costo = call_f(X[idx][0], X[idx][1], f) + Jcc(N, d, h, wa, wr, X, X[idx])
					Fi[idx][2] = costo 
					delta[0] = random.uniform(-1,1)
					delta[1] = random.uniform(-1,1)
					for m in range(Ns):
						x_gorro = (X[idx][0]+(c*delta[0]), X[idx][1]+(c*delta[1]))
						#TODO: optimizar llamadas a Jcc
						costo_gorro = call_f(x_gorro[0], x_gorro[1], f) + Jcc(N, d, h, wa, wr, X, X[idx])
						if costo_gorro < costo:
							X[idx] = x_gorro
							Fi[idx][0] = x_gorro[0]
							Fi[idx][1] = x_gorro[1]
							Fi[idx][2] = costo_gorro
						else:
							m = Ns
			#Extraer los mejores N/2 elementos y clonarlos para completar la población
			Fi.sort(key=lambda cost : cost[2])
			for i in range(int(N/2)):
				X[i] = (Fi[i][0], Fi[i][1])
			for i in range(int(N/2)):
				X[int(i+(N/2))] = (Fi[i][0], Fi[i][1])
		for idx in range (N):
			r = random.uniform(0,1)
			if r < pe:
				X[idx] = (random.uniform(xl,xu), random.uniform(xl,xu))
	#Encontrar y retornar la mejor solución del conjunto final
	minimo = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(N):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			minimo = X[idx]
			valor = valor_t
	return minimo, valor

def Jcc(N, d, h, wa, wr, X, Xi):
	resultado1 = 0
	resultado2 = 0
	for idx in range(N):
		diferencias = 0
		for m in range(2):
			diferencias = diferencias + (Xi[m]-X[idx][m])**2
		resultado1 = resultado1 + (-d*np.exp(-wa*diferencias))
		resultado2 = resultado2 + (h*np.exp(-wr*diferencias))
	return resultado1 + resultado2

def call_f(arg1, arg2, sel):
	if sel == 0:
		return sphere(arg1, arg2)
	elif sel == 1:
		return rastrigin(arg1, arg2)
	elif sel == 2:
		return ackley(arg1, arg2)
	elif sel == 3:
		return schwefel(arg1, arg2)
	else:
		return 0

rep = 30
iteraciones = []
#Sphere
resultados_z = []
resultados_x = []
resultados_y = []
for idx in range(rep):
	res, val = BFO(-5, 5, 0)
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
	res, val = BFO(-5, 5, 1)
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
	res, val = BFO(-5, 5, 2)
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

#Schwefel
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = BFO(-500, 500, 3)
	print("Schwefel (rep:", idx, "): ", res, " = ", val)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Schwefele
fig = plt.figure()
fig.suptitle('Schwefel', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-500, 500, 1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(schwefel(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()