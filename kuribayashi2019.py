import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def process_image(input_folder, output_folder, filename):
    def line_angle(line):
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        return angle

    # Create a unique directory for this file's outputs
    file_output_folder = os.path.join(output_folder, filename.split('.')[0]) + '/'
    if not os.path.exists(file_output_folder):
        os.makedirs(file_output_folder)

    # Load the original image
    original_img = cv2.imread(input_folder + filename)
    cv2.imwrite(file_output_folder + '1_original.jpg', original_img)

    # Convert the image to grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file_output_folder + '2_grayscale.jpg', gray_img)

    # Apply Gaussian blur to smooth out the noise
    gray = cv2.GaussianBlur(gray_img, (1, 1), 0)
    cv2.imwrite(file_output_folder + '3_gaussian_blur.jpg', gray)

    # Detect edges using the Canny edge detector
    edges_img = cv2.Canny(gray, 50, 150, apertureSize=3)
    cv2.imwrite(file_output_folder + '4_edges_detected.jpg', edges_img)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges_img, 1, np.pi / 180, threshold=20, minLineLength=30, maxLineGap=10)
    if lines is not None:
        # Calculate the angle of each line segment
        angles = np.array([line_angle(line) for line in lines])

        # Map labels to colors
        colors = [
            (0, 0, 255),   # Blue
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 255, 255), # Cyan
            (255, 0, 255), # Magenta
            (255, 255, 0), # Yellow
            (255, 165, 0), # Orange
            (128, 0, 128), # Purple
            (0, 0, 0)      # Black
        ]

        kmeans = KMeans(n_clusters=3)  # 3 clusters for width, height, depth
        labels = kmeans.fit_predict(angles.reshape(-1, 1))

        # Draw lines on the image with colors based on their cluster
        for line, label in zip(lines, labels):
            x1, y1, x2, y2 = line[0]
            cv2.line(original_img, (x1, y1), (x2, y2), colors[label], 1)

    cv2.imwrite(file_output_folder + '5_lines_clustered.jpg', original_img)

# Example usage:
_folder_base = 'kuribayashi2019-'
input_folder = _folder_base + 'input/'
output_folder = _folder_base + 'output/'

# List of filenames to process
filenames = ['kuribayashi-test-1.jpeg', 'kuribayashi-test-2.webp', 'kuribayashi-test-3.webp']

for filename in filenames:
    process_image(input_folder, output_folder, filename)
