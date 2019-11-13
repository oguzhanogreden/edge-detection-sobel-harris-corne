import cv2
import numpy as np
import math as m


# reads image and return it as numpy array with pixel values
def readImage(image):
    img = cv2.imread(image)
    return img


# saves Image in the working directory
def saveImage(name, image):
    cv2.imwrite(name, image)


# Finds the convolution between Image and given filter and returns new image array
def convolution(filter, gray_img):
    h, w = gray_img.shape[:2]
    offset = filter.shape[0] // 2
    new_Image = np.zeros((h, w))
    for y in range(offset, h - offset):
        for x in range(offset, w - offset):
            image_filter = gray_img[y - offset:y + offset + 1, x - offset:x + offset + 1] * filter
            conv = image_filter.sum()
            new_Image[y, x] = abs(conv)
    return new_Image


# computes the gaussian filter with given size and standard deviation value
def gaussianKernel(filter_size, sigma):
    g_filter = np.zeros((filter_size, filter_size), np.float32)
    x = filter_size // 2
    y = filter_size // 2

    grid = np.array([[((i ** 2 + j ** 2) / (2.0 * sigma ** 2)) for i in range(-x, x + 1)] for j in range(-y, y + 1)])
    g_filter = np.exp(-grid) / (2 * np.pi * sigma ** 2)
    return g_filter


# computes H-matrix of Harris detector with given parameters
# first finds determinant and traces of the matrix
# finally compare the response r value with threshold
# returns an image by marking corners
def compute_H_xy(s_xx, s_yy, s_xy,rgb_image, k, threshold):
    det = (s_xx * s_yy) - s_xy ** 2
    trace = s_xx + s_yy
    r = det - k * (trace ** 2)
    h, w = s_xx.shape[:2]
    for y in range(h):
        for x in range(w):
            if r[y, x] > threshold:
                rgb_image[y, x] = [0, 0, 255]
    return rgb_image


if __name__ == "__main__":
    # Read RGB image
    rgb_image = readImage('../data/input_hcd2.jpg')
    # convert RGB to gray scale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    # generates a gaussian filter
    gaussian_filter = gaussianKernel(filter_size=5, sigma=5)
    # Performs convolution between gaussian filter and gray image
    gaussian_image = convolution(gaussian_filter, gray_image)

    # defining horizontal and vertical filter
    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Derivative of gaussian image along x and y axes
    i_x = convolution(g_x, gaussian_image)
    i_y = convolution(g_y, gaussian_image)

    # Auto-correlation of gradient image
    i_xx = i_x * i_x
    i_yy = i_y * i_y
    # cross-correlation between x and y derivatives of image
    i_xy = i_x * i_y

    # smoothing window function with all the elements as one
    window = np.ones((3, 3), dtype=np.uint8)

    # convolution of i_xx, i_yy and i_xy with window function
    s_xx = convolution(window, i_xx)
    s_yy = convolution(window, i_yy)
    s_xy = convolution(window, i_xy)

    k = 0.06
    threshold = 10000

    # computes H matrix and obtain image with corners
    final_image = compute_H_xy(s_xx, s_yy, s_xy, rgb_image, k, threshold)

    # Save and display images
    saveImage("hcd_output.jpg", final_image)
    cv2.imshow('FinalImage:', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







