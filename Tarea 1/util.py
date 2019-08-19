import skimage.io as skio
import numpy as np


def image_read(filename, as_gray=False):
    """
    image_read: str , [bool] -> image[uint8]
    lee una imagen dada su direccion y la retorna.
    Siempre retorna una matriz de NxMx3/NxMx4/NxM en enteros sin signo de 8bits.
    :param filename: direccion para leer
    :param as_gray booleano, si es escala de grises o no.
    """
    image = skio.imread(filename, as_gray=as_gray)
    if image.dtype == np.float64:
        image = to_uint8(image)
    return image


def to_uint8(image):
    """
    Transforma una matriz de imagen a otra que sólo tiene valores
    enteros sin signo de 8 bits.
    :param image: imagen a procesar
    :return: imagen con solo enteros sin signo de 8bits
    """
    if image.dtype == np.float64:
        image = image * 255
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8, copy=False)
    return image


def image_write(filename, image):
    """
    Guarda una imagen en el direccion dada.
    :param filename: direccion para guardar
    :param image: informacion de imagen para guardar
    :return: void
    """
    skio.imsave(filename, image)


def text_to_ascii(text):
    """
    Transforma un texto a un arreglo de numeros donde cada elemento
    del arreglo es una letra del texto.
    :param text: texto a transcribir.
    :return: list(int): lista de enteros
    """
    return_list = []
    for i in text:
        return_list += [ord(i)]
    return return_list


def ascii_to_text(list):
    """
    Transforma una lista de enteros a un string
    donde cada elemento de la lista es un caracter
    del string
    :param list: lista para texto
    :return: string
    """
    return_string = ""
    for i in range(len(list)):
        return_string += chr(list[i])
    return return_string


def text_read(filename):
    """
    Lee un archivo de texto y retorna un string con el texto en este.
    :param filename: direccion del texto
    :return: texto
    """
    file = open(filename, "r")
    return_string = file.read()
    file.close()
    return return_string


def to_binary(number):
    """
    Convierte un numero a un string binario de 8 caracteres.
    :param number: Numero a convertir
    :return: String binario:
    Ej: 101 -> "1100101"
        5   -> "0000101"
    """
    binary = bin(number)[2:]  # bin retorna '0bX...X'
    while len(binary) < 8:
        binary = "0" + binary
    return binary


def to_int(string):
    """
    Convierte un string binario a un entero.
    :param string: String binario (0's y 1's)
    :return: Entero
    """
    return int(string, 2)


def last_value(number, bits):
    """
    Calcula el valor de los n-esimos bits
    menos significativos de un numero.
    :param number: Numero a calcular
    :param bits: bits menos signicativos
    :return:
    """
    return_number = to_binary(number)
    return_number = to_int(return_number[len(return_number)-bits:])
    return return_number


def join_binaries(s1, s2):
    """
    Une dos strings cortando el primero segun el tamaño del segundo.
    :param s1: string inicio
    :param s2: string final
    :return:
    ej: joins_strings("10101","11") = "10111"
    """
    return_string = s1[:len(s1)-len(s2)] + s2
    return to_int(return_string)
