import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import randint
from numpy import linspace
from mpl_toolkits import mplot3d

#Definir funciones
def sumatoria(x = []):
	resultado = 0
	for idx in range(2):
		resultado = resultado + (x[idx]-2)**2
	return resultado

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)

def drop_wave(x, y):
	return (-1) * ((1 + np.cos(12*np.sqrt(x**2+y**2)))/0.5*(x**2+y**2)+2)

def ruleta(aptitudes, aptitud_total):
	r = random.uniform(0,1)
	Psum = 0
	for idx in range(N):
		Psum = Psum + (aptitudes[idx]/aptitud_total)
		if Psum >= r:
			padre_idx = idx
			return padre_idx
	padre_idx = N - 1
	return padre_idx

def mutacion(hijos, D, xl, xu):
	pm = random.uniform(0,1)
	for idx in range(N):
		for j in range(D):
			ra = random.uniform(0,1)
			if ra < pm:
				rb = random.uniform(0,1)
				#hijos[idx][j] = xl + (xu - xl)*rb
				#Los valores separados de las tuplas son inmutables
				#por lo que es necesario reemplazar la tupla completa.
				#Esta porción del código NO funciona para D!=2
				if j == 0:
					hijos[idx] = (xl + (xu - xl)*rb, hijos[idx][1])
				else:
					hijos[idx] = (hijos[idx][0], xl + (xu - xl)*rb)
	return hijos

#Implementación del algoritmo, retorna una tupla con los mínimos encontrados (D=2)
#f encoding:
#	0: sumatoria
#	1: rastrigin
#	2: drop_wave
def GA(xl, xu, f):
	padres = []		#Lista de padres de tamaño N, cada padre es una tupla de posibles soluciones
	for idx in range(N): #Generar N padres aleatorios
		padres.append((random.uniform(xl,xu), random.uniform(xl,xu)))
	#Generaciones
	for gen in range(g):
		aptitudes = []	#Lista de aptitudes correspondientes a cada uno de los padres
		aptitud_total = 0
		for apt in range(N): #Calcular aptitud de cada padre
			eva = call_f(padres[apt][0], padres[apt][1], f)
			if eva < 0:
				aptitudes.append(1/(1+np.abs(eva)))
			else:
				aptitudes.append(1/(1+eva))
			#aptitudes.append((-1) * call_f(padres[apt][0], padres[apt][1], f))
			aptitud_total = aptitud_total + aptitudes[apt]
		hijos = [] #Conjunto vacío de hijos
		while len(hijos) < len(padres):
			#Seleccionar padres por ruleta
			padre1 = padres[ruleta(aptitudes, aptitud_total)]
			padre2 = padres[ruleta(aptitudes, aptitud_total)]
			#Generar hijos con padres seleccionados (Pc=1)
			hijo1 = (padre1[0], padre2[1])
			hijo2 = (padre2[0], padre1[1])
			hijos.append(hijo1)
			hijos.append(hijo2)
		#Mutar hijos
		hijos = mutacion(hijos, 2, xl, xu) #Dimensionalidad 2, xl -2, xu 2
		#Sustituir conjunto de padres por hijos
		padres = hijos
	#De la población final elegir la mejor solución
	#Podría ahorrarme este paso si utilizo la aptitud
	#para obtener la mejor solución, pero hacerlo de esta
	#manera me pareció más didáctico.
	minimo = padres[0]
	valor = call_f(padres[0][0], padres[0][1], f)
	for idx in range(N):
		valor_t = call_f(padres[idx][0], padres[idx][1], f)
		if(valor_t < valor):
			minimo = padres[idx]
			valor = valor_t
	return minimo, valor

def call_f(arg1, arg2, sel):
	if sel == 0:
		return sumatoria([arg1, arg2])
	elif sel == 1:
		return rastrigin(arg1, arg2)
	elif sel == 2:
		return drop_wave(arg1, arg2)
	else:
		return 0

#Algoritmo GA
#Inicialización
N = 200			#Población
g = 500			#Generaciones
rep = 30		#Número de veces que ejecutaremos el algoritmo para medir su rendimiento
iteraciones = []
#Sumatoria
resultados_sum = []
resultados_x = []
resultados_y = []
for idx in range(rep):
	res, val = GA(-2, 2, 0)
	print("Sumatoria (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	iteraciones.append(idx+1)
	resultados_sum.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar sumatoria
fig = plt.figure()
fig.suptitle('Sumatoria', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_sum)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-2, 2, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sumatoria([np.ravel(X), np.ravel(Y)]))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_sum[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()

#Rastrigin
resultados_rastrigin = []
resultados_x.clear()
resultados_y.clear()
for idx in range(rep):
	res, val = GA(-5, 5, 1)
	print("Rastrigin (rep:", idx, "): ", res, " = ", val)
	resultados_rastrigin.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar rastrigin
fig = plt.figure()
fig.suptitle('Rastrigin', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_rastrigin)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-5, 5, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_rastrigin[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()

#Drop_wave
resultados_dropwave = []
for idx in range(rep):
	res, val = GA(-5, 5, 2)
	print("Drop_Wave (rep:", idx, "): ", res, " = ", val)
	resultados_dropwave.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar drop_wave
fig = plt.figure()
fig.suptitle('Drop_Wave', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, resultados_dropwave)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(drop_wave(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_dropwave[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()