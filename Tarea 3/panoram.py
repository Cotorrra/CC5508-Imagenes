#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import random as rng

RANSAC_TRIES = 10  # intentos
INLIER_POR_THRESHOLD = 0.30  # Porcentaje de puntos Inliers para considerar la homografía válida
DISTANCE_THRESHOLD = 3 # Distancia máxima para que un punto sea inlier
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
        raise (TimeoutError("RANSAC couldnt find a fitting homography"))


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

    corners = np.array([apply_homography((0, 0), h),
                        apply_homography((src.shape[0], 0), h),
                        apply_homography((0, src.shape[1]), h),
                        apply_homography((src.shape[0], src.shape[1]), h)])

    minX, minY = np.min(corners[0, :]), np.min(corners[1, :])
    maxX, maxY = np.max(corners[0, :]), np.max(corners[1, :])

    pad = list(dst.shape)
    pad[0] = np.round(np.maximum(pad[0], maxY) - np.minimum(0, minY)).astype(int)
    pad[1] = np.round(np.maximum(pad[1], maxX) - np.minimum(0, minX)).astype(int)
    wrap_img = np.zeros(pad, dtype=np.uint8)
    trans = np.eye(3, 3)
    offsetX, offsetY = 0, 0
    if minX < 0:
        offsetX = np.round(-minX).astype(int)
        trans[0, 2] -= offsetX
    if minY < 0:
        offsetY = np.round(-minY).astype(int)
        trans[1, 2] -= offsetY
    new_h = trans.dot(h)
    new_h /= new_h[2, 2]
    wrap_img[offsetY:offsetY + dst.shape[0], offsetX:offsetX + dst.shape[1]] = dst

    inverse = np.linalg.inv(new_h)
    for j in range(0, wrap_img.shape[0]):
        for i in range(0, wrap_img.shape[1]):
            if not is_in_between((i, j), (offsetX, offsetX + dst.shape[1]), (offsetY, offsetY + dst.shape[0])):
                invert_point = apply_homography((i, j), inverse)
                try:
                    wrap_img[i, j] = src[invert_point[1], invert_point[0]]
                except Exception as e:
                    pass

    return wrap_img


def interpolate_color(color1, color2, ratio):
    new_color1 = map((lambda x: int(x * ratio)), color1)
    new_color2 = map((lambda x: int(x * (1 - ratio))), color2)
    return 1


def calculate_ratio(point, center1, center2):
    # Calcula la interpolación segun la distancia del centro de cada una de las parte
    return distance(point, center1)


def is_in_between(point, x, y):
    in_x = x[0] <= point[0] <= x[1]
    in_y = y[0] <= point[1] <= y[1]
    return in_x and in_y


def distance(p1, p2):
    return np.sqrt(np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1]))


def half_point(p):
    return int(p[0] * 0.5), int(p[1] * 0.5)


def apply_homography(point, homography):
    ext_point = np.append(point, 1)  # (x, y, 1)
    point = homography.dot(ext_point)
    point /= point[2]
    return point[0:2].astype(int)


if __name__ == "__main__":
    img_1 = cv2.imread('Images/caso_2/2a.jpg')
    img_2 = cv2.imread('Images/caso_2/2b.jpg')
    panoramic(img_1, img_2)