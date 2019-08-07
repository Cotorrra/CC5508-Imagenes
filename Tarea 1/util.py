import skimage.io as skio
import numpy as np
import basis


def image_read(filename, as_gray=False):
    """
    image_read: str , [bool] -> image[uint8]
    lee una imagen dada su direccion y la retorna.
    Siempre retorna una matriz de NxMx3/NxMx4/NxM en sin signo enteros de 8bits.
    :param filename: direccion para leer
    :param as_gray booleano, si es escala de grises o no.
    """
    image = skio.imread(filename, as_gray=as_gray)
    if image.dtype == np.float64:
        image = to_uint8(image)
    return image


def to_uint8(image):
    """
    Transforma una arreglo de imagen a uno con
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
    image_write: str -> void
    Guarda una imagen en el direccion dada.
    :param filename: direccion para guardar
    :param image: informacion de imagen para guardar
    :return: void
    """
    skio.imsave(filename, image)


def text_to_ascii(string):
    """
    Transforma un texto a un arreglo de numeros donde cada elemento
    del arreglo es una letra del texto.
    :param string: texto a transcribir.
    :return: list(int): lista de enteros
    """
    return_list = []
    for i in range(len(text)):
        return_list += [ord(text[i])]
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


def read_file(filename):
    """
    Lee un archivo de texto y retorna un string con todo el texto en este.
    :param filename: direccion del texto
    :return: texto
    """
    file = open(filename, "r")
    return_string = file.read()
    file.close()
    return return_string

