import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering


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

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)

def line_angle(line):
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle
# Calculate the angle of each line segment
angles = np.array([line_angle(line) for line in lines])

# Normalize angles to [0, 180) degrees to treat opposite directions the same
normalized_angles = np.abs(angles) % 180

# Cluster based on normalized angles
clustering = AgglomerativeClustering(n_clusters=3)  # Assuming 3 main directions: width, height, depth
labels = clustering.fit_predict(normalized_angles.reshape(-1, 1))

# Map labels to colors (fixed colors for clarity)
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green for width, height, depth

# Draw lines on the image with colors based on their cluster
for line, label in zip(lines, labels):
    x1, y1, x2, y2 = line[0]
    cv2.line(original_img, (x1, y1), (x2, y2), colors[label], 2)

cv2.imwrite('6_lines_first_clustered.jpg', original_img)