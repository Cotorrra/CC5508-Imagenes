import skimage.feature as feature
from skimage.io import ImageCollection
from skimage.io import imread
import numpy as np
import os.path
import os
import histograms


def load_as_gray(image_path):
    # Función auxiliar para leer las imagenes en grises
    return imread(image_path, as_gray=True)


def read_database():
    """
    Retorna una colección de imágenes con todas las imagenes de la base de datos.
    :return:
    """
    collection = ImageCollection('images\\BD_2\\a*\\*.jpg', conserve_memory=False, load_func=load_as_gray)
    return collection


def process_database(bins, blocks, hist_fun, canny=False):
    """
    Aplica la función de histograma a todas las imagenes de la base de datos
    :param bins:
    :param blocks:
    :param hist_fun:
    :param canny:
    :return:
    """
    database = read_database()
    hs = np.zeros(shape=(len(database), bins), dtype=np.float32)
    for i, im in enumerate(database):
        if canny:
            im = feature.canny(im, sigma=3)
        hs[i] = hist_fun(im, bins, blocks)

    return hs


def rank_images(hs, h):
    """
    Retorna las imágenes clasificadas según la distancia entre éstas y el histogramada dado.
    :param hs: Histogramas de la base de datos
    :param h: Histograma de la imagen de búsqueda
    :return: Los índices de orden de ranking de las distancias.
    """
    distances = np.zeros(len(hs), np.float32)

    for i, hist in enumerate(hs):
        distances[i] = np.sqrt(np.sum(np.square(h - hs[i])))

    return np.argsort(distances)


def sketch_retrival(image, results, hist_fun, bins, blocks, canny=False):
    """
    Recupera una cantidad especifica de imagenes según la distancia que tienen dada.
    :param image: Imagen a buscar
    :param results: Resultados
    :param hist_fun: Función de histograma
    :param bins: Cubetas para la funcion
    :param blocks: Bloques para la funcion
    :param canny: Si se usa Canny o no dentro de la función
    :return:
    """
    database = read_database()
    database_hist = process_database(bins, blocks, hist_fun, canny)
    image_hist = hist_fun(image, bins, blocks)
    index_query = rank_images(database_hist, image_hist)
    final_results = np.zeros(results, np.float32)
    for i in range(results):
        index = np.where(index_query == i)[0][0]
        final_results[i] = database[index]
    return final_results


def map_query(bins, blocks, results, hist_fun, canny=False):
    """
    Calcula el mAP de los parametros usando la query que está en images/query.txt
    :param bins: cubetas a utilizar en hist_fun
    :param blocks: bloques a utilizar en hist_fun
    :param results: resultados a pedir de la búsqueda
    :param hist_fun: función de histograma a utilizar
    :param canny: si las imágenes de búsqueda se les aplica canny o no.
    :return:
    """
    database = read_database()
    database_hist = process_database(bins, blocks, hist_fun, canny)
    total_queries = 0
    precision = 0
    with open('images/query.txt', 'r') as f:
        for line in f:
            total_queries += 1
            (key, val) = line.split()
            expected_type = val.replace("\n", "")
            image = imread('images/queries/' + key, as_gray=True)
            hist = hist_fun(image, bins, blocks)
            index_query = rank_images(database_hist, hist)

            results_types = []
            for i in range(results):
                index = np.where(index_query == i)[0][0]
                c_type = database.files[index].split('\\')
                c_type.reverse()
                results_types += [c_type[1]]

            path = 'images/BD_2/' + expected_type
            total_expected = len(os.listdir(path))

            score = 0
            rights = 0
            for i in range(results):
                if results_types[i] == expected_type:
                    rights += 1
                    score += rights / (i + 1)
            score += (total_expected - rights) / 50
            precision += score / total_expected
    return precision / total_queries


import matplotlib.pyplot as plt

if __name__ == '__main__':
    for fun in {histograms.helo, histograms.shelo}:
        for k in {36,72,96}:
            for b in {25, 12}:
                for canny in {True, False}:
                    print(str(fun)+": k="+str(k)+" B="+str(b)+" ("+str(canny)+")")
                    print(map_query(k, b, 20, fun, canny=canny))
