import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def extract_edge_points(image):
    """
    Extract white pixel (edge) points from a Canny edge-detected image.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    points = np.column_stack(np.where(edges > 0))  # (y, x) coordinates
    return points, edges

def find_best_transform_ransac(source_points, target_points):
    """
    Find the best transformation using RANSAC.
    """
    # Convert points to the correct type and shape
    source_points = np.array(source_points, dtype=np.float32)
    target_points = np.array(target_points, dtype=np.float32)

    # Ensure points are of shape (N, 2)
    if len(source_points.shape) == 1:
        source_points = source_points.reshape(-1, 2)
    if len(target_points.shape) == 1:
        target_points = target_points.reshape(-1, 2)

    # Check if there are enough points
    if len(source_points) < 4 or len(target_points) < 4:
        raise ValueError("Not enough points to find homography. Need at least 4 points.")

    # Check the shapes
    print(f"Source Points Shape: {source_points.shape}")
    print(f"Target Points Shape: {target_points.shape}")

    # Find homography matrix with RANSAC
    H, status = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)

    return H

def stitch_images(image1, image2, transform, alpha=0.5):
    """
    Stitch two images together using the computed transformation matrix with blending.
    The alpha parameter controls the blending ratio (0.0 for full image1, 1.0 for full image2).
    """
    # Get the height and width of the second image
    height, width = image2.shape[:2]
    
    # Warp image1 using the transformation matrix
    warped_image1 = cv2.warpPerspective(image1, transform, (width * 2, height))  # initially larger canvas
    
    # Resize image2 to match the size of warped_image1 (to avoid dimension mismatch)
    image2_resized = cv2.resize(image2, (warped_image1.shape[1], warped_image1.shape[0]))
    
    # Ensure both images have the same number of channels
    if len(warped_image1.shape) == 2:  # Grayscale to BGR (3 channels)
        warped_image1 = cv2.cvtColor(warped_image1, cv2.COLOR_GRAY2BGR)
    
    if len(image2_resized.shape) == 2:  # Grayscale to BGR (3 channels)
        image2_resized = cv2.cvtColor(image2_resized, cv2.COLOR_GRAY2BGR)
    
    # Perform alpha blending in the overlapping region
    blended = cv2.addWeighted(warped_image1, alpha, image2_resized, 1 - alpha, 0)
    
    # Fill the non-overlapping region with image2
    stitched = warped_image1.copy()
    stitched[:height, :width] = blended[:height, :width]
    
    # Now, crop the image to fit both images exactly by calculating the bounding box of the warped image
    # Get the bounding box of the warped image
    corners = np.array([[0, 0], [image1.shape[1], 0], [image1.shape[1], image1.shape[0]], [0, image1.shape[0]]])
    
    # Convert corners to float32 for perspectiveTransform
    corners = corners.astype(np.float32)
    
    corners_transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), transform)
    
    # Find the min and max coordinates of the transformed corners
    x_min, y_min = np.int32(corners_transformed.min(axis=0).flatten())
    x_max, y_max = np.int32(corners_transformed.max(axis=0).flatten())
    
    # Crop the stitched image to the bounding box of both images
    stitched_cropped = stitched[y_min:y_max, x_min:x_max]
    
    return stitched_cropped

# Load two overlapping images
image1 = cv2.imread("img4.png", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("img3.png", cv2.IMREAD_GRAYSCALE)

# Extract edge points from both images
points1, edges1 = extract_edge_points(image1)
points2, edges2 = extract_edge_points(image2)

def grid_sample_points(points, grid_size=50, max_points=500):
    """
    Sample points from a grid structure to reduce density in packed areas.
    """
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    grid_points = []
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)

    for x_bin in x_bins:
        for y_bin in y_bins:
            # Select points in the current grid cell
            mask = (points[:, 0] >= x_bin) & (points[:, 0] < x_bin + grid_size) & \
                   (points[:, 1] >= y_bin) & (points[:, 1] < y_bin + grid_size)
            grid_points.append(points[mask])

    # Flatten the list and limit the number of points
    grid_points = np.vstack(grid_points)
    if len(grid_points) > max_points:
        grid_points = grid_points[:max_points]

    return grid_points

# Use grid-based sampling to reduce point density
points1_grid = grid_sample_points(points1, grid_size=50, max_points=500)
points2_grid = grid_sample_points(points2, grid_size=50, max_points=500)

# Perform RANSAC with the grid-sampled points
transform = find_best_transform_ransac(points1_grid, points2_grid)

# Print the transformation matrix
print("Transformation Matrix (Homography):")
print(transform)

# Stitch the images together
stitched_image = stitch_images(image1, image2, transform)

# Visualize results
plt.figure(figsize=(15, 10))

# Show the original images with their edge points
plt.subplot(2, 2, 1)
plt.imshow(edges1, cmap="gray")
plt.scatter(points1[:, 1], points1[:, 0], color='red', s=2)  # Show points from image1
plt.title("Canny Edges (Image 1) with Points")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(edges2, cmap="gray")
plt.scatter(points2[:, 1], points2[:, 0], color='blue', s=2)  # Show points from image2
plt.title("Canny Edges (Image 2) with Points")
plt.axis("off")

# Show the stitched image
plt.subplot(2, 1, 2)
plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
plt.title("Stitched Image")
plt.axis("off")

plt.tight_layout()
plt.show()