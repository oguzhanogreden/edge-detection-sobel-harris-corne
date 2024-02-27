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
    h_filter, w_filter = filter.shape
    new_Image = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            conv = 0
            for y in range(h_filter):
                for x in range(w_filter):
                    conv = conv + (filter[y, x] * gray_img[i+y-1, j+x-1])
            new_Image[i - 1, j - 1] = abs(conv)
    return new_Image


# First it finds the convolution between image and two filters and saves the two images
# secondly finds the Gradient magnitude of the x and y derivatives of images
# finally compairing Gradient magnitude with the threshold value to detect edges
def sobel_operation(gray_image, h_filter, v_filter):
    grad_x = convolution(h_filter, gray_image)
    grad_y = convolution(v_filter, gray_image)

    h, w = gray_image.shape[:2]

    saveImage("x_comp.jpg", grad_x)
    saveImage("y_comp.jpg", grad_y)

    cv2.imshow('Gradient x:', grad_x)
    cv2.imshow('Gradient y:', grad_y)

    final_image = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            mag = m.sqrt(pow(grad_x[i, j], 2.0) + pow(grad_y[i, j], 2.0))
            if mag > 100:
                final_image[i, j] = 255
            else:
                final_image[i, j] = 0

    return final_image


if __name__ == "__main__":
    # Read RGB image file
    rgb_image = readImage('./data/image2.jpg')
    # convert RGB image to gray scale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # defining horizontal and vertical filter
    h_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    v_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Implementation of Sobel edge detection algorithm
    final_image = sobel_operation(gray_image, h_filter, v_filter)

    saveImage("sobel.jpg",final_image)
    cv2.imshow('FinalImage:', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
