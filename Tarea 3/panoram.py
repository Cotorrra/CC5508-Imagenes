#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random as rng

RANSAC_TRIES = 25  # intentos
INLIER_POR_THRESHOLD = 0.00  # Porcentaje de puntos Inliers para considerar la homografía válida
DISTANCE_THRESHOLD = 3  # Distancia máxima para que un punto sea inlier
POINTS = 4  # Se eligen 4 puntos para armar una matriz de 9x9 con 9 incognitas.


def panoramic(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)

    # Calculos en la imagen 1
    # img1 = img1.astype('uint8')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(gray1)
    kp1, des1 = sift.compute(gray1, kp1)

    # Calculos en la imagen 2
    # img2 = img2.astype('uint8')
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
        picks = [rng.randint(0, len(src) - 1) for k in range(POINTS)]
        # get points and solve the system
        src_picks = []
        dst_picks = []

        for j in range(POINTS):
            src_picks += [src[picks[j]]]  # (x, y)
            dst_picks += [dst[picks[j]]]  # (x, y)

        h = solve_homography(src_picks, dst_picks)
        # Resolver el sistema y probar qué tan buena es la "solucion"
        current_inliers = 0
        for j in range(len(src)):
            h_point = apply_homography(src[j], h)  # multiplicación de matrices, (x, y)
            if distance(h_point, dst[j]) <= DISTANCE_THRESHOLD:
                current_inliers += 1

        if current_inliers >= len(src) * INLIER_POR_THRESHOLD:
            if current_inliers >= best_inliers:
                best_inliers = current_inliers
                best_homo = h

    if best_homo is not None:
        return best_homo
    else:
        raise (TimeoutError("RANSAC couldnt find a fitting homography"))


def apply_homography(point, homography):
    ext_point = np.append(point, 1)  # (x, y, 1)
    return np.matmul(homography, ext_point)[0:2]


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
    a[-1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    h = np.linalg.solve(a, b)
    return np.reshape(h, (3, 3))


def wrap_images(img1, img2, h):
    # TODO
    # Aplicar h pixel por pixel en img2 e interpolar el color entre img1 y img2.
    wrap_img = np.zeros((img1.shape[0] + img2.shape[0], img1.shape[1] + img2.shape[1], 3))

    center = half_point(wrap_img.shape)
    img1_x = img1.shape[0]
    img1_y = img1.shape[1]

    left = int(center[0] - (img1_x / 2))
    right = int(center[0] + (img1_x / 2))
    up = int(center[1] - (img1_y / 2))
    down = int(center[1] + (img1_y / 2))

    wrap_img[left:right, up:down] = img1

    for i in img2.shape[0]:
        for j in img2.shape[1]:
            color = img2[i, j]
            h_point = apply_homography((i, j), h)
            if not is_in_between((i, j), (left, right), (up, down)):
                wrap_img[h_point[0], h_point[1]] = color

    return wrap_img


def interpolate_color(color1, color2, ratio):
    new_color1 = map((lambda x: int(x * ratio)), color1)
    new_color2 = map((lambda x: int(x * (1 - ratio))), color2)
    return 1


def is_in_between(point, x, y):
    in_x = x[0] <= point[0] <= x[1]
    in_y = y[0] <= point[1] <= y[1]
    return in_x and in_y


def distance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


def half_point(p):
    return int(p[0] % 2), int(p[1] * 0.5)


if __name__ == "__main__":
    img1 = cv2.imread('Images/caso_1/1a.jpg')
    img2 = cv2.imread('Images/caso_1/1b.jpg')
    panoramic(img1, img2)