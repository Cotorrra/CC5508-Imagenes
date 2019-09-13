
import matplotlib.pyplot as plt
import scipy.ndimage.filters as nd_filters
# import pai_io
import numpy as np

def histogram(image, bins):

    h = np.zeros(bins,np.int32)
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.transpose(sobel_x)


    s_x = nd_filters.convolve(image, sobel_x)
    s_y = nd_filters.convolve(image, sobel_y)

    angles = np.arctan2(s_x, s_y)
    angles[angles < 0] += np.pi

    magnitud = np.sqrt(np.square(s_x)+np.square(s_y))

    index = np.floor((angles/np.pi)(bins-1)) % bins

    for angle in index:
        h[h == angle] += 1
    #...

    # sobel_x = ...s
    # sobel_y = ...
    # angles = ...
    # tirar a bins.
    pass

def helo(image, bins, blocks):
    pass

def shielo(image, bins, blocks):
    pass

if __name__ == '__main__':
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    acc = np.array([0,0])
    for i in range(2):
        acc[i] = a[a%2 == i]
    print(acc)