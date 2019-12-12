

import cv2
import numpy as np


def create_morphing_video(src, dst, point_filename, images):
    # leer puntos en el fichero de puntos
    file = open(point_filename, "r")
    file_lines = file.readlines()
    lines = np.zeros((len(file_lines), 2, 4))
    for i in range(len(file_lines)):
        line = file_lines[i]
        line = line.split(" ")
        line = [int(x) for x in line[1:]]
        lines[i, 0] = line[0:4]
        lines[i, 1] = line[4:8]

    file.close()
    image_collection = morph(src, dst, lines, images)

    shape = src.shape[0:2]
    out = cv2.VideoWriter("morphing.avi",
                          cv2.VideoWriter_fourcc(*'DIVX'), 15, shape)

    for image in image_collection:
        out.write(image)
    out.release()


def perpendicular(vector):
    """
    Calcula la linea perpendicular a otra con el mismo largo
    (Esta es la función perpendicular que está en el paper)
    :param vector: Vector al cual se le calcula la perpendicular
    :return: Vector perpendicular
    """
    return np.array([vector[1], -vector[0]])


def norm(point):
    """
    Calcula la norma del punto
    :param point:
    :return:
    """
    return np.linalg.norm(point)


def calculate_uv(x, p, q):
    """
    Calcula u y v según los parametros vistos en el paper.
    :param x: punto de la imagen fuente
    :param q: punto 1 de la línea de la imagen fuente
    :param p: punto 2 de la línea de la imagen fuente
    :return: u, v Escalares
    """
    u = (x - p).dot(q - p) / (norm(q - p) ** 2)
    v = (x - p).dot(perpendicular(q - p)) / (norm(q - p))
    return u, v


def calculate_x(u, v, p_, q_):
    """
    Calcula X' según las formulas del paper de morphing
    :param u:  escalar u
    :param v:  escalar v
    :param p_: punto 1 de la linea de la imagen destino
    :param q_: punto 2 de la linea de la imagen destino
    :return: Vector
    """
    first = p_
    second = u * (q_ - p_)
    third = v * perpendicular(q_-p_)
    third /= norm(q_-p_)
    return first + second + third


def dist_point_line(point, p, q):
    """
    Calcula la distancia entre el punto dado y la recta que forma la línea
    :param point: Punto a cual se le quiere medir la distancia
    :param p: Punto 1 de la recta
    :param q: Punto 2 de la recta
    :return: Valor escalar.
    """

    # Primero calcula la ecuación de la recta
    a = -(p[1] - q[1]) / (p[0] - q[0]) # m
    b = 1
    c = p[0] * a + p[1] # n
    return (a*point[0] + b*point[1] + c) / np.linalg.norm((a, b, c))


def calculate_weight(point, p, q):
    A = 0
    B = 0
    P = 0
    dist = dist_point_line(point, p, q)
    length = norm(p-q)
    return np.power(np.divide(np.power(length, P), A + dist), B)

def morph(src, dst, lines, images):
    collection = np.array([])

    for i in range(images + 1):

        morph_image = np.zeros(src.shape)
        ratio = i / images  # 0, 1/im, 2/im, ... ,1

        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                dsum = np.array([0, 0])
                weightsum = 0
                for l in lines:
                    x = np.array([j, i]) # Dado que las imagenes se dan vuelta
                    p = np.array(l[0, 0:2])
                    q = np.array(l[0, 2:4])
                    p_ = np.array(l[1, 0:2])
                    q_ = np.array(l[1, 2:4])
                    u, v = calculate_uv(x, p, q)
                    x_ = calculate_x(u, v, p_, q_)
                    D = x - x_
                    weight = calculate_weight(x, p, q)
                    dsum += D * weight
                    weightsum += weight
                x_prime = x + (dsum / weightsum)

        # ¿Something about the points?
        # do important stuff
        # TODO implement this stuff
        morph_image = src

        collection.insert(morph_image)

    return collection


if __name__ == "__main__":
    a = ["1.", "2", "3", "4", "5", "6", "7", "8", "9"]
    a = [int(x) for x in a[1:]]
    b = np.zeros((2, 1, 4))
    b[0, 0] = a[0:4]
    b[1, 0] = a[4:8]
