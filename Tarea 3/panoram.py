
import cv2
import numpy as np
import random as rng

def panoramic(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)

    # Calculos en la imagen 1
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(gray1)
    kp1, des1 = sift.compute(gray1, kp1)

    # Calculos en la imagen 2
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
    TRIES = 10  # intentos
    INLIER_POR_THRESHOLD = 0.3  # Porcentaje de puntos Inliers para considerar la homografía válida
    DISTANCE_THRESHOLD = 3  # Distancia máxima para que un punto sea inlier
    POINTS = 4              # Se eligen 4 puntos para armar una matriz de 9x9 con 9 incognitas.

    best_inliers = 0
    best_homo = None

    for i in range(TRIES):
        picks = [rng.randint(0, len(src) - 1) for k in range(POINTS)]
        # get points and solve the system
        src_picks = []
        dst_picks = []

        for j in range(POINTS):
            src_picks += [src[picks[i]]]  # (x, y)
            dst_picks += [dst[picks[i]]]  # (x, y)

        h = solve_homography(src_picks, dst_picks)
        # Resolver el sistema y probar qué tan buena es la "solucion"
        current_inliers = 0
        for j in range(src):
            h_point = apply_homography(src[j], h)  # multiplicación de matrices, (x, y)
            if distance(h_point, dst[j]) < DISTANCE_THRESHOLD:
                current_inliers += 1

        if current_inliers > len(src) * INLIER_POR_THRESHOLD:
            if current_inliers > best_inliers:
                best_inliers = current_inliers
                best_homo = h

    if best_homo:
        return best_homo
    else:
        raise(TimeoutError("RANSAC couldnt find a fitting homography"))


def apply_homography(point, homography):
    ext_point = np.append(point, 1) # (x, y, 1)
    return np.matmul(homography, ext_point)[0:2]


def solve_homography(src_points, dst_points):
    # Se resuelve para dst = h * src
    length = len(src_points)
    a = np.zeros((2 * length + 1, 9))
    for i in range(length):
        sx = src_points[i][0]
        sy = src_points[i][1]
        dx = dst_points[i][0]
        dy = dst_points[i][1]
        a[2*i] = np.array([-sx, -sy, -1, 0, 0, 0, dx*sx, dx*sy, dx])
        a[2*i + 1] = np.array([0, 0, 0, -sx, -sy, -1, dy*sx, dy*sy, dy])
    a[-1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
    h = np.linalg.solve(a, b)
    return np.reshape(h, (3, 3))





def wrap_images(img1, img2, h):
    # TODO
    # Aplicar h pixel por pixel en img2 e interpolar el color entre img1 y img2.
    wrap_img = np.zeros((img1.shape[0]+img2.shape[0], img1.shape[1]+img2.shape[1]))

    center = half_point(img1.shape)

    wrap_img[0:img1.shape[0], 0:img1.shape[1]] = img1



    for i in img2.shape[0]:
        for j in img2.shape[1]:
            point = img2[i, j]
            h_point = apply_homography(point, h)
            if h_point < img1.shape:
                print("a")

    return cv2.warpPerspective(img1, h, (img1.shape[1] + img2.shape[1], img2.shape[0]))


def distance(p1, p2):
    return np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))


def half_point(p):
    return int(p[0] * 0.5), int(p[1] * 0.5)

if __name__ == "__main__":
    print("Hello")
    point = (100, 200)
    print(half_point(point))
