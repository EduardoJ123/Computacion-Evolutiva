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

def exponencial(x, y):
	return x * (np.exp(-x**2-y**2))

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)

def drop_wave(x, y):
	return (-1) * ((1 + np.cos(12*np.sqrt(x**2+y**2)))/0.5*(x**2+y**2)+2)

#Implementación del algoritmo, retorna una tupla con los mínimos encontrados (D=2)
#f encoding:
#	0: sumatoria
#	1: rastrigin
#	2: drop_wave
#	3: exponencial
def ES_1_mas_1(xl, xu, f):
	var_pos = 4
	padre = (random.uniform(xl,xu), random.uniform(xl,xu))
	for gen in range(g):
		r = tuple(np.random.normal(0, var_pos, 2))
		#r = (random.uniform(xl,xu), random.uniform(xl,xu))
		hijo = (padre[0]+r[0], padre[1]+r[1])
		hijo_val = call_f(hijo[0], hijo[1], f)
		padre_val = call_f(padre[0], padre[1], f)
		if hijo_val < padre_val:
			padre = hijo
			padre_val = hijo_val
	return padre, padre_val

def ES_1_mas_1_adap(xl, xu, f):
	var_pos = 4	#Definido por el usuario
	desv_es = np.sqrt(var_pos)
	c = 0.817	#Constante de contribución de incrementos y decrementos
	padre = (random.uniform(xl,xu), random.uniform(xl,xu))
	exito = 0
	for gen in range(g):
		r = tuple(np.random.normal(0, desv_es**2, 2))
		hijo = (padre[0]+r[0], padre[1]+r[1])
		hijo_val = call_f(hijo[0], hijo[1], f)
		padre_val = call_f(padre[0], padre[1], f)
		if hijo_val < padre_val:
			padre = hijo
			padre_val = hijo_val
			exito = exito + 1
		porc_ex = exito/(gen+1)
		if porc_ex < 1/5:
			desv_es = (c**2)*desv_es
		else:
			desv_es = desv_es/(c**2)
	return padre, padre_val

def ES_mu_mas_lambda(xl, xu, f):
	mu = 100
	lamb = 100
	padres = []	#Lista de padres en la forma (x, y, desv_esx, desv_esy)
	hijos = []	#Lista de hijos en la forma (x, y, desv_esx, desv_esy)
	for idx in range(mu):
		padres.append([random.uniform(xl,xu), random.uniform(xl,xu), random.uniform(0.1,xu), random.uniform(0.1,xu)])
	for gen in range(g):
		hijos.clear()
		for idx in range(lamb):
			hijo = [0,0,0,0]
			r = [0,0]
			padre1 = padres[random.randint(0,len(padres)-1)]
			padre2 = padres[random.randint(0,len(padres)-1)]
			#Recombinación sexual intermedia
			for j in range(4):
				hijo[j] = (padre1[j]+padre2[j])/2
			r[0] = float(np.random.normal(0, hijo[2]**2, 1))
			r[1] = float(np.random.normal(0, hijo[3]**2, 1))
			hijo[0] = hijo[0] + r[0]
			hijo[1] = hijo[1] + r[1]
			hijos.append(hijo)
		#Obtener los mejores mu individuos
		elementos = padres + hijos
		for i in range(len(elementos)):
			elementos[i].append(call_f(elementos[i][0], elementos[i][1], f))
		elementos.sort(key=lambda eva : eva[4])
		for i in range(len(padres)):
			padres[i] = elementos[i]
	minimo = (padres[0][0], padres[0][1])
	valor = call_f(padres[0][0], padres[0][1], f)
	for idx in range(len(padres)):
		valor_t = call_f(padres[idx][0], padres[idx][1], f)
		if(valor_t < valor):
	 		minimo = (padres[idx][0], padres[idx][1])
	 		valor = valor_t
	return minimo, valor

def ES_mu_coma_lambda(xl, xu, f):
	mu = 100
	lamb = 100
	padres = []	#Lista de padres en la forma (x, y, desv_esx, desv_esy)
	hijos = []	#Lista de hijos en la forma (x, y, desv_esx, desv_esy)
	for idx in range(mu):
		padres.append([random.uniform(xl,xu), random.uniform(xl,xu), random.uniform(0.1,xu), random.uniform(0.1,xu)])
	for gen in range(g):
		hijos.clear()
		for idx in range(lamb):
			hijo = [0,0,0,0]
			r = [0,0]
			padre1 = padres[random.randint(0,len(padres)-1)]
			padre2 = padres[random.randint(0,len(padres)-1)]
			#Recombinación sexual intermedia
			for j in range(4):
				hijo[j] = (padre1[j]+padre2[j])/2
			r[0] = float(np.random.normal(0, hijo[2]**2, 1))
			r[1] = float(np.random.normal(0, hijo[3]**2, 1))
			hijo[0] = hijo[0] + r[0]
			hijo[1] = hijo[1] + r[1]
			hijos.append(hijo)
		#Obtener los mejores mu individuos
		#al ser mu=lambda se seleccionan todos 
		#los elementos del conjunto hijos
		for i in range(len(padres)):
			padres[i] = hijos[i]
	minimo = (padres[0][0], padres[0][1])
	valor = call_f(padres[0][0], padres[0][1], f)
	for idx in range(len(padres)):
		valor_t = call_f(padres[idx][0], padres[idx][1], f)
		if(valor_t < valor):
	 		minimo = (padres[idx][0], padres[idx][1])
	 		valor = valor_t
	return minimo, valor

def call_f(arg1, arg2, sel):
	if sel == 0:
		return sumatoria([arg1, arg2])
	elif sel == 1:
		return rastrigin(arg1, arg2)
	elif sel == 2:
		return drop_wave(arg1, arg2)
	elif sel == 3:
		return exponencial(arg1, arg2)
	else:
		return 0

g = 500 		#Generaciones
rep = 30		#Número de veces que ejecutaremos el algoritmo para medir su rendimiento
resultados_x = []
resultados_y = []
resultados_z = []
iteraciones = []
#Sumatoria 1+1 ES
for idx in range(rep):
	res, val = ES_1_mas_1(-2, 2, 0)
	print("[1+1 ES]Sumatoria (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	iteraciones.append(idx+1)
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar sumatoria
fig = plt.figure()
fig.suptitle('Sumatoria', fontsize=16)
ax = fig.add_subplot(2, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES')
fig2 = plt.figure()
fig2.suptitle('Sumatoria', fontsize=24)
ax = fig2.add_subplot(2, 2, 1, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sumatoria([np.ravel(X), np.ravel(Y)]))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES')
#Sumatoria 1+1 ES Adap
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1_adap(-2, 2, 0)
	print("[1+1 ES Adap]Sumatoria (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES Adap')
ax = fig2.add_subplot(2, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sumatoria([np.ravel(X), np.ravel(Y)]))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES Adap')
#Sumatoria mu+lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_mas_lambda(-2, 2, 0)
	print("[Mu+Lambda ES]Sumatoria (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu+Lambda ES')
ax = fig2.add_subplot(2, 2, 3, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sumatoria([np.ravel(X), np.ravel(Y)]))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu+Lambda ES')
#Sumatoria Mu,Lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_coma_lambda(-2, 2, 0)
	print("[Mu,Lambda ES]Sumatoria (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 4)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu,Lambda ES')
ax = fig2.add_subplot(2, 2, 4, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sumatoria([np.ravel(X), np.ravel(Y)]))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu,Lambda ES')
plt.show()

#Exponencial 1+1 ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1(-2, 2, 0)
	print("[1+1 ES]Exponencial (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Exponencial
fig = plt.figure()
fig.suptitle('Exponencial', fontsize=16)
ax = fig.add_subplot(2, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES')
fig2 = plt.figure()
fig2.suptitle('Exponencial', fontsize=24)
ax = fig2.add_subplot(2, 2, 1, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(exponencial(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES')
#exponencial 1+1 ES Adap
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1_adap(-2, 2, 0)
	print("[1+1 ES Adap]Exponencial (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES Adap')
ax = fig2.add_subplot(2, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(exponencial(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES Adap')
#Exponencial mu+lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_mas_lambda(-2, 2, 0)
	print("[Mu+Lambda ES]Exponencial (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu+Lambda ES')
ax = fig2.add_subplot(2, 2, 3, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(exponencial(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu+Lambda ES')
#Exponencial Mu,Lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_coma_lambda(-2, 2, 0)
	print("[Mu,Lambda ES]Exponencial (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 4)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu,Lambda ES')
ax = fig2.add_subplot(2, 2, 4, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(exponencial(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu,Lambda ES')
plt.show()

#Rastrigin 1+1 ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1(-2, 2, 0)
	print("[1+1 ES]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Rastrigin
fig = plt.figure()
fig.suptitle('Rastrigin', fontsize=16)
ax = fig.add_subplot(2, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES')
fig2 = plt.figure()
fig2.suptitle('Rastrigin', fontsize=24)
ax = fig2.add_subplot(2, 2, 1, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES')
#Rastrigin 1+1 ES Adap
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1_adap(-2, 2, 0)
	print("[1+1 ES Adap]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES Adap')
ax = fig2.add_subplot(2, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES Adap')
#Rastrigin mu+lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_mas_lambda(-2, 2, 0)
	print("[Mu+Lambda ES]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu+Lambda ES')
ax = fig2.add_subplot(2, 2, 3, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu+Lambda ES')
#Rastrigin Mu,Lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_coma_lambda(-2, 2, 0)
	print("[Mu,Lambda ES]Rastrigin (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 4)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu,Lambda ES')
ax = fig2.add_subplot(2, 2, 4, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu,Lambda ES')
plt.show()

#Drop_Wave 1+1 ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1(-2, 2, 0)
	print("[1+1 ES]Drop_Wave (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
#Graficar Drop_Wave
fig = plt.figure()
fig.suptitle('Drop_Wave', fontsize=16)
ax = fig.add_subplot(2, 2, 1)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES')
fig2 = plt.figure()
fig2.suptitle('Drop_Wave', fontsize=24)
ax = fig2.add_subplot(2, 2, 1, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(drop_wave(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES')
#Drop_Wave 1+1 ES Adap
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_1_mas_1_adap(-2, 2, 0)
	print("[1+1 ES Adap]Drop_Wave (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 2)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('1+1 ES Adap')
ax = fig2.add_subplot(2, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(drop_wave(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('1+1 ES Adap')
#Drop_Wave mu+lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_mas_lambda(-2, 2, 0)
	print("[Mu+Lambda ES]Drop_Wave (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 3)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu+Lambda ES')
ax = fig2.add_subplot(2, 2, 3, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(drop_wave(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu+Lambda ES')
#Drop_Wave Mu,Lambda ES
resultados_x.clear()
resultados_y.clear()
resultados_z.clear()
for idx in range(rep):
	res, val = ES_mu_coma_lambda(-2, 2, 0)
	print("[Mu,Lambda ES]Drop_Wave (rep:", idx, "): ", res, " = ", val)
	#Crear listas para graficar
	resultados_z.append(val)
	resultados_x.append(res[0])
	resultados_y.append(res[1])
ax = fig.add_subplot(2, 2, 4)
ax.plot(iteraciones, resultados_z)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Mu,Lambda ES')
ax = fig2.add_subplot(2, 2, 4, projection='3d')
x_plot = y_plot = np.arange(-3, 3, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(drop_wave(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(resultados_x[res], resultados_y[res], resultados_z[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mu,Lambda ES')
plt.show()