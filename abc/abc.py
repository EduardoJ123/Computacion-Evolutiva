import matplotlib.pyplot as plt
import numpy as np
import random
from numpy.random import randint
from numpy import linspace
from mpl_toolkits import mplot3d

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)

def rastrigin_tras(x, y):
	return (10 * 2) + ((x+t)**2) + ((y-t)**2) - 10 * np.cos(2*np.pi*(x+t) - 10 * np.cos(2*np.pi*(y-t)))

def sphere(x, y):
	return (x + 2)**2 + (y + 2)**2

def sphere_tras(x, y):
	return ((x-t) + 2)**2 + ((y+t) + 2)**2

def ackley(x, y):
	return (-20)*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + 20 + np.exp(1)

def ackley(x, y):
	return (-20)*np.exp(-0.2*np.sqrt(0.5*((x+t)**2+(y+t)**2))) - np.exp(0.5*(np.cos(2*np.pi*(x+t))+np.cos(2*np.pi*(y+t)))) + 20 + np.exp(1)

def ABC(xl, xu, f):
	L = 100		#Número máximo de intentos - definido por el usuario
	Pf = 100	#Número de abejas empleadas
	Po = 100	#Número de abejas observadoras
	X = []		#Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	T = []		#Lista de contador de intentos, cada uno asociado a un elemento de X
	#Etapa de inicialización
	for i in range(N): #Inicializar aleatoriamente fuentes de comida y contadores a 0
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
		T.append(0)
	#Generaciones
	for gen in range(g):
		#Etapa de abejas empleadas
		for i in range(Pf):
			k = rand_diff(0,Pf-1,i)
			j = random.randint(0,1)	#TODO: generalizar para D!=2
			phi = random.uniform(-1,1)
			v_i = X[i]
			if j == 0:	#TODO: generalizar para D!=2. Por ahora solo funciona para D==2
				v_ij = (X[i][j]+phi*(X[i][j]-X[k][j]), v_i[1])
			else:
				v_ij = (v_i[0], X[i][j]+phi*(X[i][j]-X[k][j]))
			if call_f(v_ij[0], v_ij[1], f) < call_f(X[i][0], X[i][1], f):
				X[i] = v_ij
				T[i] = 0
			else:
				T[i] = T[i] + 1
		#Etapa de abejas observadoras
		for i in range(Po):
			aptitudes = []	#Lista de aptitudes correspondientes a cada uno de los padres
			aptitud_total = 0
			for apt in range(N):
				eva = call_f(X[apt][0], X[apt][1], f)
				if eva < 0:
					aptitudes.append(1/(1+np.abs(eva)))
				else:
					aptitudes.append(1/(1+eva))
				aptitud_total = aptitud_total + aptitudes[apt]
			m = ruleta(aptitudes, aptitud_total)
			k = rand_diff(1,Pf-1,m)
			j = random.randint(0,1)
			phi = random.uniform(-1,1)
			v_m = X[m]
			if j == 0:	#TODO: generalizar para D!=2. Por ahora solo funciona para D==2
				v_mj = (X[m][j]+phi*(X[m][j]-X[k][j]), v_m[1])
			else:
				v_mj = (v_m[0], X[m][j]+phi*(X[m][j]-X[k][j]))
			if call_f(v_mj[0], v_mj[1], f) < call_f(X[m][0], X[m][1], f):
				X[m] = v_mj
				T[m] = 0
			else:
				T[m] = T[m] + 1
		#Etapa de abejas exploradoras
		for i in range(Pf):
			if T[i] > L:
				X[i] = (random.uniform(xl,xu), random.uniform(xl,xu))
				T[i] = 0
	#Encontrar y retornar la mejor solución del conjunto final
	minimo = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(N):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			minimo = X[idx]
			valor = valor_t
	return minimo, valor

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

def rand_diff(inf, sup, dif):
	flag = 1
	while flag:
		n = random.randint(inf,sup)
		if n != dif:
			flag = 0
	return n

def call_f(arg1, arg2, sel):
	if sel == 0:
		return sphere(arg1, arg2)
	elif sel == 1:
		return rastrigin(arg1, arg2)
	elif sel == 2:
		return ackley(arg1, arg2)
	elif sel == 3:
		return sphere_tras(arg1, arg2)
	elif sel == 4:
		return rastrigin_tras(arg1, arg2)
	elif sel == 5:
		return ackley_tras(arg1, arg2)
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
t = random.randint(1,100)
for idx in range(rep):
	res, val = ABC(-5, 5, 0)
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
	res, val = ABC(-5, 5, 1)
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
	res, val = ABC(-5, 5, 2)
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