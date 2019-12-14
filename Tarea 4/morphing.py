
import cv2
from util import *
import time

# Constantes para el calculo de peso
A = 1
B = 1
P = 0.5
# (length^p / (A + dist))^ b


def create_morphing_video(src, dst, point_filename, n_images):
    # leer puntos en el fichero de puntos
    file = open(point_filename, "r")
    file_lines = file.readlines()
    lines_src = np.zeros((len(file_lines), 4))
    lines_dst = np.zeros((len(file_lines), 4))
    for i in range(len(file_lines)):
        line = file_lines[i]
        line = line.split(" ")
        line = [int(x) for x in line[1:]]
        lines_src[i] = line[0:4]
        lines_dst[i] = line[4:8]

    file.close()
    src = src.astype('uint8')
    dst = dst.astype('uint8')
    morph(src, dst, lines_src, lines_dst, n_images)


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
                x = np.array([j, i])  # Dado que las imagenes se dan vuelta
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
            x_prime = (x + (dsum / weightsum)).astype(int)
            if 0 <= x_prime[1] < img_src.shape[0] and 0 <= x_prime[0] < img_src.shape[1]:
                result[i, j] = img_src[x_prime[1], x_prime[0]]
    return result


def morph(src, dst, lines_src, lines_dst, n_images):

    shape = src.shape[0:2]
    out = cv2.VideoWriter("morphing.avi",
                          cv2.VideoWriter_fourcc(*'DIVX'), 10, shape)

    for i in range(n_images):
        t = 1 - np.divide(i, (n_images - 1))  # 0, 1/im, 2/im, ... , 1
        sd_lines = interpolate_lines(lines_src, lines_dst, t)
        ds_lines = interpolate_lines(lines_dst, lines_src, t)
        wrap_s = wrap(src, lines_src, sd_lines)
        wrap_d = wrap(dst, lines_dst, ds_lines)
        morph_image = t*wrap_s + (1-t)*wrap_d
        morph_image = morph_image.astype('uint8')
        cv2.imwrite("Tests/img"+str(i)+".png", morph_image)
        out.write(morph_image)
        print("Image number " + str(i+1) + " is done, with t = "+str(np.round(t,2))+" ("+str(int(-100*(t-1)))+"%)")

    out.release()


if __name__ == "__main__":
    img1 = cv2.imread("Caras/couple0.jpg")
    img2 = cv2.imread("Caras/couple1.jpg")
    line_file = "Caras/lines.txt"
    start_time = time.process_time()
    create_morphing_video(img1, img2, line_file, 100)
    print("--- %.2f seconds ---" % (time.process_time() - start_time))
