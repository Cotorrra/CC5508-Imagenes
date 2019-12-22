import numpy as np
import os

A = 1
B = 1
P = 0.5

def interpolate_lines(lines1, lines2, t):
    """
    Interpola dos lineas segun t
    :param lines1: set de lineas 1
    :param lines2: set de lineas 2
    :param t: escalar entre 0 y 1
    :return:
    """
    return t * lines1 + (1 - t) * lines2


def itos(number):
    """
    Calcula el string segun el numero, donde los numeros de 1 digito tienen 2 numeros
    ie: 2 -> 02
    :param number: numero a convertir
    :return: 
    """
    if number < 10:
        return "0" + str(number)
    else:
        return str(number)


def perpendicular(vector):
    """
    Calcula la linea perpendicular a otra con el mismo largo
    (Esta es la función perpendicular que está en el paper)
    :param vector: Vector al cual se le calcula la perpendicular
    :return: Vector perpendicular
    """
    return np.array([vector[1], -vector[0]])


def norm(p):
    """
    Calcula la norma del punto
    :param p:
    :return:
    """
    return np.sqrt(np.power(p[0], 2) + np.power(p[1], 2))


def dot_product(p1, p2):
    """
    Calcula el producto punto entre dos puntos
    :param p1:
    :param p2:
    :return:
    """
    return p1[0] * p2[0] + p1[1] * p2[1]


def calculate_uv(x, p, q):
    """
    Calcula u y v según los parametros vistos en el paper.
    :param x: punto de la imagen fuente
    :param q: punto 1 de la línea de la imagen fuente
    :param p: punto 2 de la línea de la imagen fuente
    :return: u, v Escalares
    """
    u = dot_product(x - p, q - p) / np.power(norm(q - p), 2)
    v = dot_product(x - p, perpendicular(q - p)) / (norm(q - p))
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
    third = (v * perpendicular(q_ - p_)) / norm(q_ - p_)
    return first + second + third


def dist_point_line(point, u, v, p, q):
    """
    Calcula la distancia entre el punto dado y la recta que forma la línea
    :param point: Punto a cual se le quiere medir la distancia
    :param u: valor de u actual
    :param v: valor de v actual
    :param p: Punto 1 de la recta
    :param q: Punto 2 de la recta
    :return: Valor escalar.
    """
    if 0 < u < 1:
        return np.abs(v)
    elif u < 0:
        return norm(point - p)
    else:
        return norm(point - q)


def calculate_weight(point, u, v, p, q):
    dist = dist_point_line(point, u, v, p, q)
    length = norm(p - q)
    return np.power(np.divide(np.power(length, P), A + dist), B)
