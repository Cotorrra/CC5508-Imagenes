import numpy as np


def interpolate_lines(lines1, lines2, t):
    """
    Interpola dos lineas segun t
    :param lines1: set de lineas 1
    :param lines2: set de lineas 2
    :param t: escalar entre 0 y 1
    :return:
    """
    return t*lines1 + (1-t)*lines2


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
    u = (x - p).dot(q - p) / np.power(norm(q - p), 2)
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
        return norm(point-p)
    else:
        return norm(point-q)


def calculate_weight(point, u, v, p, q, A, B, P):
    dist = dist_point_line(point, u, v, p, q)
    length = norm(p-q)
    return np.power(np.divide(np.power(length, P), A + dist), B)
