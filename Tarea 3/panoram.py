#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random as rng

RANSAC_TRIES = 30  # intentos
INLIER_POR_THRESHOLD = 0.30  # Porcentaje de puntos Inliers para considerar la homografía válida
DISTANCE_THRESHOLD = 4 # Distancia máxima para que un punto sea inlier
POINTS = 4  # Se eligen 4 puntos para armar una matriz de 9x9 con 9 incognitas.
DEBUG = False


def panoramic(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)
    # Calculos en la imagen 1
    img1 = img1.astype('uint8')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(gray1)
    kp1, des1 = sift.compute(gray1, kp1)

    # Calculos en la imagen 2
    img2 = img2.astype('uint8')
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2 = sift.detect(gray2)
    kp2, des2 = sift.compute(gray2, kp2)

    # Encontrar concordancias entre los descriptores locales de las dos imagenes
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    # Calculando homografía con las correspondencias
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        h = find_homography_ransac(src_pts, dst_pts)

    # Aplicar la transformación correspondiente
    dst = wrap_images(img1, img2, h)
    cv2.imwrite("panoramica.png", dst)


def find_homography_ransac(src, dst):
    best_inliers = 0
    best_homo = None

    for i in range(RANSAC_TRIES):
        picks = rng.sample(range(len(src)), POINTS)
        src_picks = []
        dst_picks = []
        for j in range(POINTS):
            src_picks += [src[picks[j]]]  # (x, y)
            dst_picks += [dst[picks[j]]]  # (x, y)

        # Resolver el sistema
        h = solve_homography(src_picks, dst_picks)

        current_inliers = 0
        for j in range(len(src)):
            h_point = apply_homography(src[j], h)  # multiplicación de matrices, (x, y)
            if distance(h_point, dst[j]) <= DISTANCE_THRESHOLD:
                current_inliers = current_inliers + 1

        if current_inliers >= len(src) * INLIER_POR_THRESHOLD:
            if current_inliers >= best_inliers:
                print(current_inliers / len(src))
                best_inliers = current_inliers
                best_homo = h

    if best_homo is not None:
        return best_homo
    else:
        raise TimeoutError("RANSAC couldnt find a fitting homography")


def solve_homography(src_points, dst_points):
    # Se resuelve para dst = h * src
    a = np.zeros((2 * POINTS + 1, 9))
    for i in range(POINTS):
        sx = src_points[i][0]
        sy = src_points[i][1]
        dx = dst_points[i][0]
        dy = dst_points[i][1]
        a[2 * i] = np.array([-sx, -sy, -1, 0, 0, 0, dx * sx, dx * sy, dx])
        a[(2 * i) + 1] = np.array([0, 0, 0, -sx, -sy, -1, dy * sx, dy * sy, dy])
    a[8] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    try:
        h = np.linalg.solve(a, b)
    except Exception as e:
        # Con SIFT algunas coordenadas quedan repetidas
        # Por lo que hay que atrapar los errores de solve
        h = b

    return np.reshape(h, (3, 3))


def wrap_images(src, dst, h):
    # TODO
    # Aplicar h pixel por pixel en img2 e interpolar el color entre img1 y img2.
    wrap_img = np.zeros((dst.shape[0], src.shape[0] + dst.shape[1], 3))
    wrap_img[0:dst.shape[0], 0:dst.shape[1]] = dst
    inverse = np.linalg.inv(h)
    # inverse /= inverse[2, 2]
    for i in range(0, wrap_img.shape[0]):
        for j in range(0, wrap_img.shape[1]):
            if is_null_color(wrap_img[i, j]):
                point = apply_homography((j, i), inverse)
                try:
                    wrap_img[i, j] = interpolate_color(point, src)
                except IndexError as e:
                    pass

    return wrap_img


def is_null_color(color):
    return (color[0] == 0) and (color[1] == 0) and (color[2] == 0)


def interpolate_color(point, image):
    base_x = np.floor(point[0])
    ratio_x = point[0] - base_x
    base_y = np.floor(point[1])
    ratio_y = point[1] - base_y
    return_color = interpolate_component((1-ratio_x)*(1-ratio_y), image[point[1], point[0]])
    return_color += interpolate_component(ratio_x*(1-ratio_y), image[point[1] + 1, point[0]])
    return_color += interpolate_component(ratio_y * (1 - ratio_x), image[point[1], point[0] + 1])
    return_color += interpolate_component(ratio_x * ratio_y, image[point[1] + 1, point[0] + 1])
    return return_color


def interpolate_component(ratio, component):
    return np.array([ratio * component[0], ratio * component[1], ratio * component[2]]).astype(int)


def distance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


def apply_homography(point, homography):
    ext_point = np.append(point, 1)  # (x, y, 1)
    point = np.matmul(homography, ext_point)
    point /= point[2]
    return point[0:2].astype(int)


if __name__ == "__main__":
    img_1 = cv2.imread('Images/caso_4/img08.jpg')
    img_2 = cv2.imread('Images/caso_4/img01.jpg')
    panoramic(img_1, img_2)
