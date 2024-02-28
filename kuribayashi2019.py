import cv2
import numpy as np

# Step 1: Load the original image
original_img = cv2.imread('kuribayashi-test-2.webp')
cv2.imwrite('1_original.jpg', original_img)

# Step 2: Convert the image to grayscale
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('2_grayscale.jpg', gray_img)

# Step 3: Enhance contrast using histogram equalization
# If the image is grayscale, use equalizeHist directly
# equalized_img = cv2.equalizeHist(gray_img)
# cv2.imwrite('3_contrast_enhanced.jpg', equalized_img)

# Apply Gaussian blur to smooth out the noise
gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
cv2.imwrite('3_gaussian_blur.jpg', gray)

# Step 4: Detect edges using the Canny edge detector
edges_img = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imwrite('4_edges_detected.jpg', edges_img)

# Step 5: Use Hough Transform to detect lines
lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)

# Step 6: Draw lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

cv2.imwrite('5_lines_drawn.jpg', original_img)
