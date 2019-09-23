
import matplotlib.pyplot as plt
import scipy.ndimage.filters as nd_filters
# import pai_io
from skimage.util.shape import view_as_blocks
import numpy as np


def convolve_sobel(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.transpose(sobel_x)
    g_x = nd_filters.convolve(image, sobel_x)
    g_y = nd_filters.convolve(image, sobel_y)
    return g_x, g_y


def histogram(image, bins):
    h = np.zeros(bins, np.float32)
    g_x, g_y = convolve_sobel(image)

    angles = np.arctan2(g_x, g_y)
    angles[angles < 0] += np.pi

    magnitude = np.sqrt(np.square(g_x)+np.square(g_y))

    index = np.floor((angles/np.pi)(bins-1)) % bins

    for i in range(bins):
        rows, cols = np.where(index == i)
        h[i] = np.sum(magnitude[rows, cols])
    h = h / np.linalg.norm(h,2)

    return h


def helo(image, bins, block_size):

    blocks_x = np.ceil(image.shape[0] / block_size)
    blocks_y = np.ceil(image.shape[1] / block_size)

    g_x, g_y = convolve_sobel(image)
    magnitude = np.sqrt(np.square(g_x) + np.square(g_y))

    G_x = np.square(g_x) - np.square(g_y)
    G_y = 2 * g_x * g_y
    L_x = np.zeros(shape=(blocks_x,blocks_y))
    L_y = np.zeros(shape=(blocks_x, blocks_y))

    for i in range(blocks_x):
        rows = range(i, min(i + block_size,image.shape[0]))
        for j in range(blocks_y):
            cols = range(j, max(j + block_size,image.shape[1]))
            L_x[i, j] = np.sum(G_x[rows, cols])
            L_y[i, j] = np.sum(G_y[rows, cols])

    b_angles = 0.5 * np.arctan2(L_x, L_y)
    b_angles[b_angles < 0] += np.pi / 2

    # interpolacion lineal

    return h

    pass


def shielo(image, bins, blocks):
    image_blocks = view_as_blocks(image, (blocks, blocks))

    pass


def numbers(a,b):
    return a, b


if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    a[a < 6] += 5
    r, c = np.where(a == 6)
    print(r)
    print(c)
    print(a)
    c, d = numbers(1, 2)
    print(c)
    print(d)

