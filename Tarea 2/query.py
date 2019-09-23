

import skimage.feature as feature
from skimage.io import imread_collection
import numpy as np

def orientations(image, hist):
    pass


def apply_canny(image):
    return feature.canny(image, sigma=3)


def query(image, bins, blocks, results, query_fun, canny=False):

    database = imread_collection('image/DB_2/*.jpg')
    histograms = np.zeros(shape=(len(database), bins))
    distances = np.zeros(len(database), np.float32)

    for i, im in enumerate(database):
        histograms[i] = query_fun(im)

    if canny:
        image = apply_canny(image)

    image_histogram = query_fun(image)

    for i, hist in enumerate(histograms):
        distances[i] = np.sqrt(np.sum(np.square(image_histogram - histograms[i])))

    # sacar los resultados mayores

    return

