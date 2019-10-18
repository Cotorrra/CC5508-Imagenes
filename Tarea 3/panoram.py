
import cv2
import numpy as np


def panoramic(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=6, sigma=1.6)

    # make calculations on image1
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kp1 = sift.detect(gray1)
    kp1, des1 = sift.compute(gray1, kp1)

    # make calculations on image2
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2 = sift.detect(gray2)
    kp2, des2 = sift.compute(gray2, kp2)

    # match the multiple local descriptors
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    ##Calculando homografía para afinar correspondencias
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        matchesMask = mask.ravel().tolist()
    else:
        matchesMask = None

    # Tengo el match con sift

    img3 = 1 # Imagen irl


    #

