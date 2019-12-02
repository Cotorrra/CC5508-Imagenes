#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 11:39:28 2018

@author: jsaavedr
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img_ = cv2.imread('../images/stitching/caso_3/3b.jpg')
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img1)
    kp, des = sift.compute(img1, kp)

    img = cv2.imread('../images/stitching/caso_3/3a.jpg')
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp_2 = sift.detect(img2)
    kp_2, des_2 = sift.compute(img2, kp_2)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des, des_2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    ##Calculando homografía para afinar correspondencias
    if len(good) > 4:
        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst_pts = np.float32([kp_2[m.trainIdx].pt for m in good]).reshape(-1, 2)
        src_xs = src_pts[:, 0]
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        matchesMask = mask.ravel().tolist()
        print(M)
    else:
        matchesMask = None

    dst = cv2.warpPerspective(img_, M, (img_.shape[1] + img.shape[1], img.shape[0]))
    plt.subplot(122)
    plt.imshow(dst)
    plt.title("WarpedImage")
    plt.show()
    plt.figure()
    dst[0:img.shape[0], 0:img.shape[1]] = img
    cv2.imwrite("output.jpg", dst)
    plt.imshow(dst)
    plt.show()

    #
    # Aquí, matchesMask contiene las correspondencias
    # img3 = cv2.drawMatches(img, kp, img_2, kp_2, good, None, flags=2, matchesMask=matchesMask)

    # cv2.imshow("matches", img3)
    # cv2.waitKey()
