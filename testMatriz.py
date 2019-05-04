import numpy
import sys
import math

matriz1 = numpy.zeros((3,3))
matriz2 = numpy.zeros((3,1))


matrizA = numpy.matrix( [
                        [math.cos(math.radians(5)), 0, math.sin(math.radians(5))],
                        [0, 1, 0],
                        [math.sin(math.radians(5)) * -1 , 0, math.cos(math.radians(5))]
                        ]
                        )

matrizB = numpy.matrix([[1],[2],[3]])

matrizB_regreso = numpy.matrix ([[1.25766193], [2], [2.90142835]])

matrizA_regreso = numpy.matrix( [
                        [math.cos(math.radians(-5)), 0, math.sin(math.radians(-5))],
                        [0, 1, 0],
                        [math.sin(math.radians(-5)) * -1 , 0, math.cos(math.radians(-5))]
                        ]
                        )

print (matriz1)
print (matriz2)
print ("********")

print ("matrizA")
print (matrizA)

print ("********")

print ("matrizB")
print (matrizB)


print ("********")
print ("resultado con * ")
resultado = matrizA * matrizB

print (resultado)


print ("*********")
print ("resultado con numpy.dot() ")

resultado2 = numpy.dot(matrizA, matrizB)

print (resultado2)

print ("//////")
a = float(resultado2[0])
b = float(resultado2[1])
c = float(resultado2[2])

print (a)
print (b)
print (c)

print ("*********")
print ("*********")
print ("*********")
print ("*********")
print ("*********")

resultado3 = numpy.dot(matrizA_regreso, matrizB_regreso)
print (resultado3)

