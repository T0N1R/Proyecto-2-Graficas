import pygame
from OpenGL.GL import *
import OpenGL.GL.shaders as shaders
import glm
import pyassimp
import numpy
import math
import time

# pygame

pygame.init()
pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
clock = pygame.time.Clock()
pygame.key.set_repeat(1, 10)


glClearColor(0.18, 0.18, 0.18, 1.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_TEXTURE_2D)

# shaders

vertex_shader = """
#version 440
layout (location = 0) in vec4 position;
layout (location = 1) in vec4 normal;
layout (location = 2) in vec2 texcoords;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec4 color;
uniform vec4 light;

out vec4 vertexColor;
out vec2 vertexTexcoords;

void main()
{
    float intensity = dot(normal, normalize(light - position));

    gl_Position = projection * view * model * position;
    vertexColor = color * intensity;
    vertexTexcoords = texcoords;
}

"""

fragment_shader = """
#version 440
layout (location = 0) out vec4 diffuseColor;

in vec4 vertexColor;
in vec2 vertexTexcoords;

uniform sampler2D tex;

void main()
{
    diffuseColor = vertexColor * texture(tex, vertexTexcoords);
}
"""

shader = shaders.compileProgram(
    shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
    shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER),
)
glUseProgram(shader)


# matrixes
model = glm.mat4(1)
view = glm.mat4(1)
projection = glm.perspective(glm.radians(45), 800/600, 0.1, 1000.0)

glViewport(0, 0, 800, 600)


scene = pyassimp.load('./models/OBJ/torreTokio.obj')


def glize(node, var):
    model = node.transformation.astype(numpy.float32)

    for mesh in node.meshes:
        material = dict(mesh.material.properties.items())
        texture = material['file'][2:]

        if var == 1:
            texture_surface = pygame.image.load("./models/OBJ/" + texture)
            texture_data = pygame.image.tostring(texture_surface,"RGB",1)
            width = texture_surface.get_width()
            height = texture_surface.get_height()
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)

        if var == 2:
            texture_surface = pygame.image.load("./models/OBJ/engineflare1.jpg")
            texture_data = pygame.image.tostring(texture_surface,"RGB",1)
            width = texture_surface.get_width()
            height = texture_surface.get_height()
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)


        if var == 3:
            texture_surface = pygame.image.load("./models/OBJ/drkwood2.jpg")
            texture_data = pygame.image.tostring(texture_surface,"RGB",1)
            width = texture_surface.get_width()
            height = texture_surface.get_height()
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
            glGenerateMipmap(GL_TEXTURE_2D)




        vertex_data = numpy.hstack((
            numpy.array(mesh.vertices, dtype=numpy.float32),
            numpy.array(mesh.normals, dtype=numpy.float32),
            numpy.array(mesh.texturecoords[0], dtype=numpy.float32)
        ))

        faces = numpy.hstack(
            numpy.array(mesh.faces, dtype=numpy.int32)
        )

        vertex_buffer_object = glGenVertexArrays(1)
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        glVertexAttribPointer(0, 3, GL_FLOAT, False, 9 * 4, None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, False, 9 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)


        element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

        glUniformMatrix4fv(
            glGetUniformLocation(shader, "model"), 1 , GL_FALSE, 
            model
        )
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "view"), 1 , GL_FALSE, 
            glm.value_ptr(view)
        )
        glUniformMatrix4fv(
            glGetUniformLocation(shader, "projection"), 1 , GL_FALSE, 
            glm.value_ptr(projection)
        )

        diffuse = mesh.material.properties["diffuse"]

        glUniform4f(
            glGetUniformLocation(shader, "color"),
            *diffuse,
            1
        )

        glUniform4f(
            glGetUniformLocation(shader, "light"), 
            -100, 300, 100, 1
        )

        glDrawElements(GL_TRIANGLES, len(faces), GL_UNSIGNED_INT, None)


    for child in node.children:
        glize(child, var)

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje y
matrizY = numpy.matrix( [
                        [math.cos(math.radians(10)), 0, math.sin(math.radians(10))],
                        [0, 1, 0],
                        [math.sin(math.radians(10)) * -1 , 0, math.cos(math.radians(10))]
                        ]
                        )

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje y
matrizY_regreso = numpy.matrix( [
                        [math.cos(math.radians(-10)), 0, math.sin(math.radians(-10))],
                        [0, 1, 0],
                        [math.sin(math.radians(-10)) * -1 , 0, math.cos(math.radians(-10))]
                        ]
                        )

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje x
matrizX = numpy.matrix([
                        [1, 0, 0],
                        [0, math.cos(math.radians(10)), math.sin(math.radians(10)) * -1],
                        [0, math.sin(math.radians(10)), math.cos(math.radians(10))]
                        ]
                        )

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje x
matrizX_regreso = numpy.matrix([
                        [1, 0, 0],
                        [0, math.cos(math.radians(-10)), math.sin(math.radians(-10)) * -1],
                        [0, math.sin(math.radians(-10)), math.cos(math.radians(-10))]
                        ]
                        )

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje z
matrizZ = numpy.matrix([
                        [math.cos(math.radians(10)), math.sin(math.radians(10)) * -1, 0],
                        [math.sin(math.radians(10)), math.cos(math.radians(10)), 0],
                        [0,0,1]
                        ]
                        )

#matriz que al multiplicarse con los valores x, y z de la camara, va a mover la imagen 10 grados alrededor del eje z
matrizZ_regreso = numpy.matrix([
                        [math.cos(math.radians(-10)), math.sin(math.radians(-10)) * -1, 0],
                        [math.sin(math.radians(-10)), math.cos(math.radians(-10)), 0],
                        [0,0,1]
                        ]
                        )



camera = glm.vec3(0, 0, 30)
print(camera.x)
print(camera.y)
print(camera.z)
var = 1
#camera_speed = 50

"""
def process_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                camera.x += camera_speed
                camera.z += camera_speed
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")
                
            if event.key == pygame.K_RIGHT:
                camera.x -= camera_speed
                camera.z -= camera_speed
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")
    return False
"""

def process_input(var):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
            return True
        if event.type == pygame.KEYDOWN:

            #Girar a la izquierda (girar sobre el eje Y)
            if event.key == pygame.K_LEFT:

                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizY_regreso, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                                
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")


            #Girar a la derecha (girar sobre el eje Y)
            if event.key == pygame.K_RIGHT:

                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizY, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")


            #Girar hacia arriba (girar sobre el eje X)
            if event.key == pygame.K_UP:

                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizX, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")


            #Girar hacia abajo (girar sobre el eje X)
            if event.key == pygame.K_DOWN:
                
                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizX_regreso, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")

            #Zoom in al oprimir la tecla z
            if event.key == pygame.K_z:
                camera.z -= 1
                
                if camera.z <= 10:
                    camera.z = 10
                
                    
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")
                

            #Zoom out al oprimir la tecla x
            if event.key == pygame.K_x:
                camera.z += 1
                if camera.z > 30:
                    camera.z = 30

                
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")

            #Girar alrededor del eje Z al presionar "i"
            if event.key == pygame.K_i:
                
                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizZ, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")

            #Girar alrededor del eje Z al presionar "o"
            if event.key == pygame.K_o:
                
                matrizB = numpy.matrix([[camera.x],[camera.y],[camera.z]])

                resultado = numpy.dot(matrizZ_regreso, matrizB)

                camera.x = float(resultado[0])
                camera.y = float(resultado[1])
                camera.z = float(resultado[2])
                print("******************************")
                print(camera.x)
                print(camera.y)
                print(camera.z)
                print("******************************")

            if event.key == pygame.K_u:
                var = 2

            if event.key == pygame.K_h:
                var = 3


            if event.key == pygame.K_i:
                var = 1
                
    return False, var



done = False
while not done:
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    view = glm.lookAt(camera, glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

    glize(scene.rootnode, var)

    done, var = process_input(var)
    clock.tick(15)
    pygame.display.flip()
