import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy import linspace
from mpl_toolkits import mplot3d

dx = 0.1
dy = 0.1
h = 0.01
n = 500

iteraciones = []
x_l = []
y_l = []
z_l = []

#Definir funciones
def sphere(x, y):
	return (x + 2)**2 + (y + 2)**2

def rastrigin(x, y):
	return (10 * 2) + (x**2) + (y**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y)

print("Algoritmo para la esfera")
for rep in range(10):
	x = randint(10)
	y = randint(10)
	for idx in range(n):
		#Calcular el gradiente
		#Diferencias finitas para x
		x_p = x + (dx / 2)
		x_n = x - (dx / 2)
		f_p = (x_p + 2)**2 + (y + 2)**2		
		f_n = (x_n + 2)**2 + (y + 2)**2
		dpx = (f_p - f_n)/dx
		#Diferencias finitas para y
		y_p = y + (dy / 2)
		y_n = y - (dy / 2)
		f_p = (x + 2)**2 + (y_p + 2)**2
		f_n = (x + 2)**2 + (y_n + 2)**2
		dpy = (f_p - f_n)/dy
		grad = dpx + dpy
		#Actualizar valores
		x = x - (h * dpx)
		y = y - (h * dpy)
		z = sphere(x, y)
		#Imprimimos resultados de cada iteración
		print("\niter= ", str(idx+1), "\nx = ", str(x), "\ny = ", str(y), "\nz = ", str(z))
	#Guardamos solo resultados finales para graficar
	x_l.append(x)
	y_l.append(y)
	z_l.append(z)
	iteraciones.append(rep+1)		

#Graficar
fig = plt.figure()
fig.suptitle('Sphere', fontsize=24)
ax = fig.add_subplot(1, 2, 1)
ax.plot(iteraciones, z_l)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-20, 20, .1)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(sphere(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(x_l[res], y_l[res], z_l[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.5)
ax.set_title('Mínimo')
plt.show()
res_esfera = z_l.copy()

print("Algoritmo para rastrigin")
iteraciones.clear()
x_l.clear()
y_l.clear()
z_l.clear()
for rep in range(10):
	x = randint(10)
	y = randint(10)
	for idx in range(n):
		#Calcular el gradiente
		#Diferencias finitas para x
		x_p = x + (dx / 2)
		x_n = x - (dx / 2)
		f_p = (10 * 2) + (x_p**2) + (y**2) - 10 * np.cos(2*np.pi*x_p) - 10 * np.cos(2*np.pi*y)
		f_n = (10 * 2) + (x_n**2) + (y**2) - 10 * np.cos(2*np.pi*x_n) - 10 * np.cos(2*np.pi*y)
		dpx = (f_p - f_n)/dx
		#Diferencias finitas para y
		y_p = y + (dy / 2)
		y_n = y - (dy / 2)
		f_p = (10 * 2) + (x**2) + (y_p**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y_p)
		f_n = (10 * 2) + (x**2) + (y_n**2) - 10 * np.cos(2*np.pi*x) - 10 * np.cos(2*np.pi*y_n)
		dpy = (f_p - f_n)/dy
		grad = dpx + dpy
		#Actualizar valores
		x = x - (h * dpx)
		y = y - (h * dpy)
		z = sphere(x, y)
		#Imprimimos resultados de cada iteración
		print("\niter= ", str(idx+1), "\nx = ", str(x), "\ny = ", str(y), "\nz = ", str(z))
	#Guardamos solo resultados finales para graficar
	x_l.append(x)
	y_l.append(y)
	z_l.append(z)
	iteraciones.append(rep+1)	

#Graficar
fig2 = plt.figure()
fig2.suptitle('Rastrigin', fontsize=24)
ax = fig2.add_subplot(1, 2, 1)
ax.plot(iteraciones, z_l)
ax.set_xlabel('No. de ejecución')
ax.set_ylabel('Mínimo')
ax.set_title('Resultados')
ax = fig2.add_subplot(1, 2, 2, projection='3d')
x_plot = y_plot = np.arange(-6, 6, .01)
X, Y = np.meshgrid(x_plot, y_plot)
zs = np.array(rastrigin(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
for res in range(len(iteraciones)):
	ax.scatter(x_l[res], y_l[res], z_l[res], marker="x", c="red", alpha=1, linewidths=2)
ax.plot_surface(X, Y, Z, alpha=0.2)
ax.set_title('Mínimo')
plt.show()
res_rastrigin = z_l.copy()

print("Varianza resultados esfera: ", np.var(res_esfera))
print("Media de resultados esfera: ", np.mean(res_esfera))
print("Varianza resultados rastrigin: ", np.var(res_rastrigin))
print("Media de resultados rastrigin: ", np.mean(res_rastrigin))