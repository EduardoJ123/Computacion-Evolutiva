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

#TODO: 	refactorizar funciones en la forma
#		DE(xl,xu,f): algoritmo general 
#		vec_mut(sel): obtención del vector de mutación
def DE_rand_1_bin(xl, xu, f):
	F = 0.5		#Tamaño de paso (factor de escalamiento) - definido por el usuario
	Cr = 0.7	#Tasa de crossover (constante de recombinación) - definido por el usiario
	X = []		#Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	V = []		#Vector de mutación
	U = []		#Vector de prueba
	#Inicializar N individuos aleatoriamente
	for i in range(N):
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
	for gen in range(g):
		V.clear()
		for i in range(N):
			xr1 = X[random.randint(0,len(X)-1)]
			xr2 = X[random.randint(0,len(X)-1)]
			xr3 = X[random.randint(0,len(X)-1)]
			op1 = (xr2[0]-xr3[0], xr2[1]-xr3[1])
			op2 = (F*op1[0], F*op1[1])
			op3 = (xr1[0]+op2[0], xr1[1]+op2[1])
			V.append(op3)
			u_temp = [0,0]	#Inicializar lista de prueba
			for j in range(2):	#Dimensión = 2 TODO: Generalizar
				ra = random.uniform(0,1)
				if ra <= Cr:
					u_temp[j] = V[i][j]
				else:
					u_temp[j] = X[i][j]
			u_temp = tuple(u_temp)
			eval_u = call_f(u_temp[0], u_temp[1], f)
			eval_x = call_f(X[i][0], X[i][1], f)
			if eval_u < eval_x:
				X[i] = u_temp
	#Encontrar y retornar la mejor solución del conjunto final
	minimo = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(N):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			minimo = X[idx]
			valor = valor_t
	return minimo, valor

def DE_best_1_bin(xl, xu, f):
	F = 0.5		#Tamaño de paso (factor de escalamiento) - definido por el usuario
	Cr = 0.7	#Tasa de crossover (constante de recombinación) - definido por el usiario
	X = []		#Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	V = []		#Vector de mutación
	U = []		#Vector de prueba
	#Inicializar N individuos aleatoriamente
	for i in range(N):
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
	for gen in range(g):
		V.clear()
		for i in range(N):
			xb = find_best(X, f)
			xr2 = X[random.randint(0,len(X)-1)]
			xr3 = X[random.randint(0,len(X)-1)]
			op1 = (xr2[0]-xr3[0], xr2[1]-xr3[1])
			op2 = (F*op1[0], F*op1[1])
			op3 = (xb[0]+op2[0], xb[1]+op2[1])
			V.append(op3)
			u_temp = [0,0]	#Inicializar lista de prueba
			for j in range(2):	#Dimensión = 2 TODO: Generalizar
				ra = random.uniform(0,1)
				if ra <= Cr:
					u_temp[j] = V[i][j]
				else:
					u_temp[j] = X[i][j]
			u_temp = tuple(u_temp)
			eval_u = call_f(u_temp[0], u_temp[1], f)
			eval_x = call_f(X[i][0], X[i][1], f)
			if eval_u < eval_x:
				X[i] = u_temp
	#Encontrar y retornar la mejor solución del conjunto final
	minimo = X[0]
	valor = call_f(X[0][0], X[0][1], f)
	for idx in range(N):
		valor_t = call_f(X[idx][0], X[idx][1], f)
		if(valor_t < valor):
			minimo = X[idx]
			valor = valor_t
	return minimo, valor

def DE_target_to_best_1_bin(xl, xu, f):
	F = 0.5		#Tamaño de paso (factor de escalamiento) - definido por el usuario
	Cr = 0.7	#Tasa de crossover (constante de recombinación) - definido por el usiario
	X = []		#Lista de posibles soluciones, cada elemento de la lista es una tupla de la forma (x, y)
	V = []		#Vector de mutación
	U = []		#Vector de prueba
	#Inicializar N individuos aleatoriamente
	for i in range(N):
		X.append((random.uniform(xl,xu), random.uniform(xl,xu)))
	for gen in range(g):
		V.clear()
		for i in range(N):
			xb = find_best(X, f)
			xr2 = X[random.randint(0,len(X)-1)]
			xr3 = X[random.randint(0,len(X)-1)]
			op1 = (xb[0]-X[i][0], xb[1]-X[i][1])
			op2 = (op1[0]+xr2[0], op1[1]+xr2[1])
			op3 = (op2[0]-xr3[0], op2[1]-xr3[1])
			op4 = (F*op3[0], F*op3[1])
			op5 = (X[i][0]+op4[0], X[i][1]+op4[1])
			V.append(op5)
			u_temp = [0,0]	#Inicializar lista de prueba
			for j in range(2):	#Dimensión = 2 TODO: Generalizar
				ra = random.uniform(0,1)
				if ra <= Cr:
					u_temp[j] = V[i][j]
				else:
					u_temp[j] = X[i][j]
			u_temp = tuple(u_temp)
			eval_u = call_f(u_temp[0], u_temp[1], f)
			eval_x = call_f(X[i][0], X[i][1], f)
			if eval_u < eval_x:
				X[i] = u_temp
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

#Algoritmo DE
#Inicialización
N = 200		#Tamaño de la población
g = 500		#Número de generaciones
rep = 30	#Número de veces que se ejecutará el algoritmo
iteraciones = []
#Sphere
resultados_z = []
resultados_x = []
resultados_y = []
for idx in range(rep):
	res, val = DE_rand_1_bin(-5, 5, 0)
	print("[DE/rand/1/bin]Sphere (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	iteraciones.append(idx+1)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar sphere
fig = plt.figure()
fig.suptitle('Sphere', fontsize=16)
ax = fig.add_subplot(1, 3, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/rand/1/bin')
fig2 = plt.figure()
fig2.suptitle('Sphere', fontsize=24)
ax = fig2.add_subplot(1, 3, 1, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sphere(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/rand/1/bin')
#Sphere DE/best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_best_1_bin(-5, 5, 0)
	print("[DE/best/1/bin]Sphere (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/best/1/bin')
ax = fig2.add_subplot(1, 3, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sphere(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/best/1/bin')
#Sphere DE/target-to-best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_target_to_best_1_bin(-5, 5, 0)
	print("[DE/target-to-best/1/bin]Sphere (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/target-to-best/1/bin')
ax = fig2.add_subplot(1, 3, 3, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sphere(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/target-to-best/1/bin')
plt.show()

#Rastrigin
resultados_z.clear()
resultados_x.clear()
resultados_y.clear()
for idx in range(rep):
	res, val = DE_rand_1_bin(-5, 5, 1)
	print("[DE/rand/1/bin]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Rastrigin
fig = plt.figure()
fig.suptitle('Rastrigin', fontsize=16)
ax = fig.add_subplot(1, 3, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/rand/1/bin')
fig2 = plt.figure()
fig2.suptitle('Rastrigin', fontsize=24)
ax = fig2.add_subplot(1, 3, 1, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/rand/1/bin')
#Rastrigin DE/best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_best_1_bin(-5, 5, 1)
	print("[DE/best/1/bin]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/best/1/bin')
ax = fig2.add_subplot(1, 3, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/best/1/bin')
#Rastrigin DE/target-to-best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_target_to_best_1_bin(-5, 5, 1)
	print("[DE/target-to-best/1/bin]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/target-to-best/1/bin')
ax = fig2.add_subplot(1, 3, 3, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/target-to-best/1/bin')
plt.show()

#Ackley
resultados_z.clear()
resultados_x.clear()
resultados_y.clear()
for idx in range(rep):
	res, val = DE_rand_1_bin(-5, 5, 2)
	print("[DE/rand/1/bin]Ackley (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Ackley
fig = plt.figure()
fig.suptitle('Ackley', fontsize=16)
ax = fig.add_subplot(1, 3, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/rand/1/bin')
fig2 = plt.figure()
fig2.suptitle('Ackley', fontsize=24)
ax = fig2.add_subplot(1, 3, 1, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(ackley(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/rand/1/bin')
#Ackley DE/best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_best_1_bin(-5, 5, 2)
	print("[DE/best/1/bin]Ackley (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/best/1/bin')
ax = fig2.add_subplot(1, 3, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(ackley(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/best/1/bin')
#Ackley DE/target-to-best/1/bin
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = DE_target_to_best_1_bin(-5, 5, 2)
	print("[DE/target-to-best/1/bin]Ackley (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(1, 3, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('DE/target-to-best/1/bin')
ax = fig2.add_subplot(1, 3, 3, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(ackley(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('DE/target-to-best/1/bin')
plt.show()