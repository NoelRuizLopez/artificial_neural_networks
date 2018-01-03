#!/usr/bin/python
# NumPy is a general-purpose array-processing package designed to efficiently 
# manipulate large multi-dimensional arrays of arbitrary records without 
# sacrificing too much speed for small multi-dimensional arrays. 
#
# NumPy is built on the Numeric code base and adds features introduced by 
# numarray as well as an extended C-API and the ability to create arrays of 
# arbitrary type which also makes NumPy suitable for interfacing with 
# general-purpose data-base applications.
#
# There are also basic facilities for discrete fourier transform, 
# basic linear algebra and random number generation.
#
# All numpy wheels distributed from pypi are BSD licensed.
# Manual: https://docs.scipy.org/doc/numpy/index.html
import numpy


# Creo una mini red neuronal para un XOR logico tipo

##
# Al final la RN es:
#     * Una capa de entradas
#     * Las conexiones de TODAS las entradas con TODAS las salidas
#     * Una capa de salidas
# La red neuronal se encarga de computar las entradas con las conexiones 
# para predecir las salidas
#
# Para que sea realmente util necesitamos saber cuanto afecta cada conexion
# a la salida

# Para empezar generamos una matriz de 2x4 tipo tabla de verdad
#  0  0
#  0  1
#  1  0
#  1  1
entrada_x = numpy.array([[0,0],[0,1],[1,0],[1,1]])
print "entrada:\n", entrada_x

#Luego la matriz de los resultados esperados, que para una XOR seria:
# agregamos el .T porque lo queremos en formato columna
salida_esperada = numpy.array([[0,1,1,0]]).T
print "Salida esperada:\n", salida_esperada

#Resultado: tabla de verdad de la operacion XOR
print "RESULT:\n", numpy.append(entrada_x, salida_esperada, axis=1)

# Para aprender la RN debe computar su prediccion, calcular el error, 
# corregir los pesos y volver a computar
# Como tenemos DOS ENTRADAS y una SALIDA, lo que buscamos es conectar
# cada ENTRADA con la SALIDA, cada ENTRADA debe tener un peso, es decir
# cada Entrada modifica mas o menos la SALIDA segun su peso

# Esta es la funcion de activacion que nos permite modelar problemas
# NO LINEALES. Esta mapea cualquier valor entre 0 y 1
# Aprovechamos la misma funcion para devolver su derivada,

def sigmoid(x, deriv=False):
	if deriv:
		ret= x*(1-x)
	else:
		ret= 1/(1+numpy.exp(-x))

	return ret

# Definimos ahora una SEMILLA que nos ayudara a que los numeros esten
# distribuidos de forma aleatoria, pero siempre igual para entender 
# en que afectan los cambios que realicemos
numpy.random.seed(0)

# Inicializamos las conexiones con una media a 0
# Esta es la matriz que ira "aprendiendo"
# La matriz resultado es de 2x1 (Dos entradas, 1 salida)

# Creamos matriz de 2x3 para las conexiones de la ENTRADA a CAPA1
conection_0 = 2*numpy.random.random((2,3)) -1

# Y las conexiones de la CAPA1 a la SALIDA: de 3 a 1 unica Salida
conection_1 = 2+numpy.random.random((3,1)) -1

# Preparamos ahora interacciones para aprender. El rango es modificable 
# para estudiar cambios
for i in xrange(70000):
	# Primero, computamos con nuestra red (forward)
        # que resultado tenemos con los pesos.
        #Inicializamos random
	l0 = entrada_x

	# Multiplicamos las entradas por las conexiones
	l1 = sigmoid(numpy.dot(l0, conection_0))
	# En l1 tendriamos los resultados de la primera activacion

	# Movemos estos datos a la siguiente capa
	l2 = sigmoid(numpy.dot(l1, conection_1))

	# Ahora computamos el error de la capa final
	l2_error = salida_esperada - l2

	# Computamos la diferencia con la derivada, que nos da el
 	# valor que debemos agregar a los pesos para "aprender"
	l2_delta = l2_error * sigmoid(l2, True)

	# Ahora debemos saber que parte del error acumulado es por culpa
	# de las primeras conexiones
	l1_error = numpy.dot(l2_delta, conection_1.T)

	# Para terminar computamos que valor debemos ajustar en 
	# las conexiones conection_1
	l1_delta = l1_error * sigmoid(l1, True)

 	# Las primeras conexiones las corregimos con l1_delta * entradas
	conection_0 += numpy.dot(l0.T, l1_delta)

 	# Las segundas conexiones las corregimos con l2_delta * entradas
	# de la capa intermedia
	conection_1 += numpy.dot(l1.T, l2_delta)
	if (i% 1000) == 0:
		print "Error:" + str(numpy.mean(numpy.abs(l2_error)))


print "\nLos pesos finales de la movida"
print l2



