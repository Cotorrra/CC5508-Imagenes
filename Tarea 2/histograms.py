import matplotlib.pyplot as plt
import scipy.ndimage.filters as nd_filters
from matplotlib import patches
import numpy as np
import skimage.io as skio


def convolve_sobel(image):
    """
    Retorna las matrices gradiente calculadas usando Sobel
    :param image:
    :return:
    """
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.transpose(sobel_x)
    g_x = nd_filters.convolve(image.astype(np.float32), sobel_x)
    g_y = nd_filters.convolve(image.astype(np.float32), sobel_y)
    return g_x, g_y


def histogram(image, bins, block):
    h = np.zeros(bins, np.float32)
    g_x, g_y = convolve_sobel(image)

    angles = np.arctan2(g_y, g_x)
    angles[angles < 0] += np.pi

    magnitude = np.sqrt(np.square(g_x) + np.square(g_y))

    index = np.floor((angles / np.pi) * (bins - 1)) % bins

    for i in range(bins):
        rows, cols = np.where(index == i)
        h[i] = np.sum(magnitude[rows, cols])
    h = h / np.linalg.norm(h, 2)

    return h


def helo(image, bins, block):
    height = image.shape[0]
    width = image.shape[1]

    g_x, g_y = convolve_sobel(image)
    L_x = np.zeros(shape=(block + 1, block + 1))
    L_y = np.zeros(shape=(block + 1, block + 1))

    for i in range(height):
        for j in range(width):
            r = round((i / height) * block)
            s = round((j / width) * block)
            L_x[r, s] += np.square(g_x[i, j]) - np.square(g_y[i, j])
            L_y[r, s] += 2 * g_x[i, j] * g_y[i, j]

    b_angles = 0.5 * np.arctan2(L_y, L_x)
    b_angles[b_angles < 0] += np.pi / 2
    magnitude = np.sqrt(np.square(g_x) + np.square(g_y))
    index = np.floor((b_angles / np.pi) * (bins - 1)) % bins
    h = np.zeros(bins, np.float32)

    for i in range(bins):
        rows, cols = np.where(index == i)
        h[i] = np.sum(magnitude[rows, cols])
    h = h / np.linalg.norm(h, 2)

    return h


def shelo(image, bins, block):
    height = image.shape[0]
    width = image.shape[1]
    g_x, g_y = convolve_sobel(image)
    L_x = np.zeros(shape=(block + 1, block + 1))
    L_y = np.zeros(shape=(block + 1, block + 1))

    for i in range(height):
        for j in range(width):
            r = (i * block) / height
            s = (j * block) / width
            left = round(s - 0.5)
            right = min(left + 1, block - 1)
            bot = round(r - 0.5)
            top = min(bot + 1, block - 1)
            d_l = s - left - 0.5
            if d_l < 0.5:
                w_l = 0.5 - d_l
            else:
                w_l = d_l - 0.5
            w_r = 1 - w_l

            d_b = r - bot - 0.5
            if d_b < 0.5:
                w_b = 0.5 - d_b
            else:
                w_b = d_b - 0.5
            w_t = 1 - w_b

            L_x[left, top] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_l * w_t
            L_y[left, top] += (2 * g_x[i, j] * g_y[i, j]) * w_l * w_t

            L_x[right, bot] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_r * w_b
            L_y[right, bot] += (2 * g_x[i, j] * g_y[i, j]) * w_r * w_b

            L_x[left, bot] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_b * w_l
            L_y[left, bot] += (2 * g_x[i, j] * g_y[i, j]) * w_l * w_b

            L_x[right, top] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_r * w_t
            L_y[right, top] += (2 * g_x[i, j] * g_y[i, j]) * w_r * w_t

    b_angles = 0.5 * np.arctan2(L_y, L_x)
    b_angles[b_angles < 0] += np.pi / 2

    index = np.floor((b_angles / np.pi) * (bins - 1)) % bins

    h = np.zeros(bins, np.float32)

    magnitude = np.sqrt(np.square(g_x) + np.square(g_y))

    for i in range(bins):
        rows, cols = np.where(index == i)
        h[i] = np.sum(magnitude[rows, cols])
    h = h / np.linalg.norm(h, 2)

    return h


def graph_std(image, bins):
    h = histogram(image, bins, 0)
    fig, xs = plt.subplots(1, 2)
    xs[0].set_axis_off()
    xs[0].imshow(image, cmap='gray')
    xs[1].bar(x=range(bins), height=h)
    plt.show()


def graph_helo(image, block):
    height = image.shape[0]
    width = image.shape[1]

    g_x, g_y = convolve_sobel(image)
    L_x = np.zeros(shape=(block + 1, block + 1))
    L_y = np.zeros(shape=(block + 1, block + 1))

    for i in range(height):
        for j in range(width):
            r = round((i / height) * block)
            s = round((j / width) * block)
            L_x[r, s] += np.square(g_x[i, j]) - np.square(g_y[i, j])
            L_y[r, s] += 2 * g_x[i, j] * g_y[i, j]

    b_angles = 0.5 * np.arctan2(L_y, L_x)
    b_angles[b_angles < 0] += np.pi / 2

    block_size_x = round((width / block)-0.5)
    block_size_y = round((height / block)-0.5)
    fig, ax = plt.subplots(1)
    ax.set_axis_off()
    ax.imshow(image, cmap='gray')
    for r in range(block):
        for s in range(block):
            angle = b_angles[r, s]
            height = 0.5 * block_size_y
            center_x = (s * block_size_x) + block_size_x / 2
            center_y = (r * block_size_y) + block_size_y / 2
            top_x = center_x + height * np.cos(angle)
            top_y = center_y + height * np.sin(angle)
            bot_x = center_x - height * np.cos(angle)
            bot_y = center_y - height * np.sin(angle)
            ax.add_patch(patches.FancyArrowPatch((top_x, top_y), (bot_x, bot_y), arrowstyle="-", color='y'))

    plt.show()


def graph_shelo(image, block):
    height = image.shape[0]
    width = image.shape[1]
    g_x, g_y = convolve_sobel(image)
    L_x = np.zeros(shape=(block + 1, block + 1))
    L_y = np.zeros(shape=(block + 1, block + 1))

    for i in range(height):
        for j in range(width):
            r = (i * block) / height
            s = (j * block) / width
            left = round(s - 0.5)
            right = min(left + 1, block - 1)
            bot = round(r - 0.5)
            top = min(bot + 1, block - 1)
            d_l = s - left - 0.5
            if d_l < 0.5:
                w_l = 0.5 - d_l
            else:
                w_l = d_l - 0.5
            w_r = 1 - w_l

            d_b = r - bot - 0.5
            if d_b < 0.5:
                w_b = 0.5 - d_b
            else:
                w_b = d_b - 0.5
            w_t = 1 - w_b

            L_x[left, top] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_l * w_t
            L_y[left, top] += (2 * g_x[i, j] * g_y[i, j]) * w_l * w_t

            L_x[right, bot] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_r * w_b
            L_y[right, bot] += (2 * g_x[i, j] * g_y[i, j]) * w_r * w_b

            L_x[left, bot] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_b * w_l
            L_y[left, bot] += (2 * g_x[i, j] * g_y[i, j]) * w_l * w_b

            L_x[right, top] += (np.square(g_x[i, j]) - np.square(g_y[i, j])) * w_r * w_t
            L_y[right, top] += (2 * g_x[i, j] * g_y[i, j]) * w_r * w_t

    b_angles = 0.5 * np.arctan2(L_y, L_x)
    b_angles[b_angles < 0] += np.pi / 2
    block_size_x = round((width / block) - 0.5)
    block_size_y = round((height / block) - 0.5)
    fig, ax = plt.subplots(1)
    ax.set_axis_off()
    ax.imshow(image, cmap='gray')
    for r in range(block):
        for s in range(block):
            angle = b_angles[r, s]
            height = 0.5 * block_size_y
            center_x = (s * block_size_x) + block_size_x / 2
            center_y = (r * block_size_y) + block_size_y / 2
            top_x = center_x + height * np.cos(angle)
            top_y = center_y + height * np.sin(angle)
            bot_x = center_x - height * np.cos(angle)
            bot_y = center_y - height * np.sin(angle)
            ax.add_patch(patches.FancyArrowPatch((top_x, top_y), (bot_x, bot_y), arrowstyle="-", color='y'))

    plt.show()


if __name__ == '__main__':
    im = skio.imread("../Tarea 2/images/birb.png", as_gray=True)
    graph_shelo(im, 50)
    graph_helo(im, 30)
