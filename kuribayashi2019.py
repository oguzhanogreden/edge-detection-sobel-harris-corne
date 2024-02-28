import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.spatial.distance import pdist, squareform


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
gray = cv2.GaussianBlur(gray_img, (1, 1), 0)
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

# Map labels to colors (fixed colors for clarity)
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

# Map labels to colors
colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Example: Blue for width, Red for height, Green for depth

# Draw lines on the image with colors based on their cluster
for line, label in zip(lines, labels):
    x1, y1, x2, y2 = line[0]
    cv2.line(original_img, (x1, y1), (x2, y2), colors[label], 1)

cv2.imwrite('6_lines_first_clustered.jpg', original_img)

# Cluster based on normalized angles
clustering = AgglomerativeClustering(n_clusters=3)  # Assuming 3 main directions: width, height, depth
labels = clustering.fit_predict(normalized_angles.reshape(-1, 1))

# Function to calculate the perpendicular distance from a point to a line segment
def point_to_line_distance(point, line):
    # Assuming line is given by two points (x1, y1) and (x2, y2)
    x1, y1, x2, y2 = line
    x0, y0 = point
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = ((y2 - y1)**2 + (x2 - x1)**2)**0.5
    if denominator == 0:
        return 0  # To avoid division by zero
    return numerator / denominator

# Adjusted function to calculate distances
def calculate_distances(lines):
    # Generate all line midpoint coordinates
    midpoints = [[(line[0][0] + line[0][2]) / 2.0, (line[0][1] + line[0][3]) / 2.0] for line in lines]
    
    # Use pdist to calculate the pairwise distances between line midpoints
    # If you need a custom metric, define it and pass to pdist via the 'metric' argument
    distance_vector = pdist(midpoints, metric='euclidean')  # Example using Euclidean, replace as needed

    return squareform(distance_vector)

# Assuming 'labels' from the first clustering are available
# For simplicity, this example does not separate lines by first clustering labels before calculating distances
distance_matrix = calculate_distances(lines)

# Second clustering based on the custom distance matrix
clustering2 = AgglomerativeClustering(n_clusters=9, metric="precomputed", linkage="complete")
labels2 = clustering2.fit_predict(distance_matrix)

# Draw lines on the image with color based on their second cluster
for line, label2 in zip(lines, labels2):
    x1, y1, x2, y2 = line[0]
    color = colors[label2 % len(colors)]  # Choose color based on second label
    cv2.line(original_img, (x1, y1), (x2, y2), color, 2)

cv2.imwrite('7_lines_second_clustered.jpg', original_img)