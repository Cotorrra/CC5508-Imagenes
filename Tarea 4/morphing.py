import cv2
from util import *
import time

# Constantes para el calculo de peso
A = 1
B = 1
P = 0.5
# (length^p / (A + dist))^ b


def create_morphing_video(src, dst, point_filename, n_images, DEBUG=False, SAVE_IM=False):
    file = open(point_filename, "r")
    file_lines = file.readlines()
    lines_src = np.zeros((len(file_lines), 4))
    lines_dst = np.zeros((len(file_lines), 4))

    for i in range(len(file_lines)):
        if DEBUG:
            print("Reading Lines... "+str(int(100 * (i + 1) / (len(file_lines) + 1)))+"%")
        line = file_lines[i]
        line = line.split(" ")
        line = [int(x) for x in line[1:]]
        lines_src[i] = line[0:4]
        lines_dst[i] = line[4:8]

    if DEBUG:
        print("Starting Morphing...")

    file.close()
    src = src.astype('uint8')
    dst = dst.astype('uint8')
    collection = morph(src, dst, lines_src, lines_dst, n_images, DEBUG, SAVE_IM)

    if DEBUG:
        print("Morphing is complete!")

    shape = (src.shape[1], src.shape[0])
    out = cv2.VideoWriter("morphing.avi",
                          cv2.VideoWriter_fourcc(*'XVID'), 1, shape)

    if DEBUG:
        print("Creating video...")

    for image in collection:
        out.write(image)

    out.release()


def wrap(img_src, lines_src, lines_dst):
    """
    Implementa el wrap de una imagen con inverse mapping.
    :param img_src: imagen fuente
    :param lines_src: lineas de correspondencia con la fuente
    :param lines_dst: lineas de correspondencia con el destino
    :return: wrapped_image
    """
    result = np.zeros(img_src.shape)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            dsum = np.array([0, 0], dtype=float)
            weightsum = 0
            for k in range(lines_src.shape[0]):
                x = np.array([j, i])
                p = np.array(lines_dst[k, 0:2])
                q = np.array(lines_dst[k, 2:4])
                p_ = np.array(lines_src[k, 0:2])
                q_ = np.array(lines_src[k, 2:4])
                u, v = calculate_uv(x, p, q)
                x_ = calculate_x(u, v, p_, q_)
                d = x_ - x
                weight = calculate_weight(x, u, v, p, q, A, B, P)
                dsum += d * weight
                weightsum += weight
            x_prime = x + (dsum / weightsum)
            if 0 <= x_prime[1] < img_src.shape[0] - 1 and 0 <= x_prime[0] < img_src.shape[1] - 1:
                # Interpolacion bilineal del color de la imagen
                left = int(np.floor(x_prime[0]))
                h_ratio = x_prime[0] - left
                up = int(np.floor(x_prime[1]))
                v_ratio = x_prime[1] - up
                result[i, j] = (1 - h_ratio) * (1 - v_ratio) * img_src[up, left] + \
                               (1 - h_ratio) * v_ratio * img_src[up + 1, left] + \
                               h_ratio * (1 - v_ratio) * img_src[up, left + 1] + \
                               h_ratio * v_ratio * img_src[up + 1, left + 1]
            else:
                # Se interpola el color segun los vecinos ya calculados
                if i == 0 and j == 0:
                    result[i, j] = img_src[i, j]
                elif j == 0 and i > 0:
                    result[i, j] = result[i - 1, j]
                elif j > 0 and i == 0:
                    result[i, j] = result[i, j - 1]
                elif 0 < j < result.shape[1] - 1 and i > 0:
                    result[i, j] = ((result[i, j - 1] / 4)
                                    + (result[i - 1, j - 1] / 4)
                                    + (result[i - 1, j] / 4)
                                    + (result[i - 1, j + 1] / 4)).astype('uint8')
                elif j > 0 and j == result.shape[1] - 1 and i > 0:
                    result[i, j] = ((result[i, j - 1] / 3)
                                    + (result[i - 1, j - 1] / 3)
                                    + (result[i - 1, j] / 3)).astype('uint8')
    return result


def morph(src, dst, lines_src, lines_dst, n_images, DEBUG=False, SAVE_IM=False):
    arr = []
    for i in range(n_images):
        t = np.divide(i, (n_images - 1))
        lines_sd = interpolate_lines(lines_src, lines_dst, t)
        lines_ds = interpolate_lines(lines_dst, lines_src, t)
        wrap_s = wrap(src, lines_src, lines_sd)
        wrap_d = wrap(dst, lines_dst, lines_ds)
        morph_image = (1 - t) * wrap_s + t * wrap_d
        morph_image = morph_image.astype('uint8')
        arr.append(morph_image)
        if SAVE_IM:
            cv2.imwrite("Tests/img" + itos(i + 1) + ".png", morph_image)
        if DEBUG:
            print("Processing Images... "+str(int(100 * (i + 1) / n_images)) + "%")

    return arr


if __name__ == "__main__":
    mode = False
    if mode:
        img1 = cv2.imread("Figuras/cl.jpg")
        img2 = cv2.imread("Figuras/sq.jpg")
        line_file = "Figuras/lines.txt"
    else:
        img1 = cv2.imread("Caras/couple0.jpg", cv2.IMREAD_COLOR)
        img2 = cv2.imread("Caras/couple1.jpg", cv2.IMREAD_COLOR)
        line_file = "Caras/lines.txt"

    start_time = time.process_time()
    create_morphing_video(img1, img2, line_file, 10, DEBUG=True, SAVE_IM=True)
    print("--- %.2f seconds ---" % (time.process_time() - start_time))
