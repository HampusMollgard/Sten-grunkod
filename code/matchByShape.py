import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the two images
image1_path = 'img1.png'  # Replace with the path to your first image
image2_path = 'img2.png'  # Replace with the path to your second image

# Read the images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

if image1 is None or image2 is None:
    print("Error loading images!")
    exit()

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get binary images (invert the threshold to get black objects)
_, mask1 = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY_INV)  # Inverted threshold for black objects
_, mask2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY_INV)  # Inverted threshold for black objects

# Find contours of the two images
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the largest contour from each image (now black objects)
contour1 = max(contours1, key=cv2.contourArea)
contour2 = max(contours2, key=cv2.contourArea)

# Draw the contours on the original images
image1_with_contours = image1.copy()
image2_with_contours = image2.copy()
cv2.drawContours(image1_with_contours, [contour1], -1, (0, 255, 0), 2)  # Green contour
cv2.drawContours(image2_with_contours, [contour2], -1, (0, 255, 0), 2)  # Green contour

# Compare the shapes of the contours using cv2.matchShapes
shape_similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)
print(f"Shape similarity: {shape_similarity}")

# Initialize stitched_image to be the first image (default if not stitching happens)
stitched_image = np.copy(image1)

# Align and stitch the second image with the first if shapes are similar
if shape_similarity < 0.5:  # Adjust the threshold as needed
    # Calculate the moments for the contours to find the centroids
    moments1 = cv2.moments(contour1)
    moments2 = cv2.moments(contour2)

    # Calculate centroids
    center1 = (int(moments1["m10"] / moments1["m00"]), int(moments1["m01"] / moments1["m00"]))
    center2 = (int(moments2["m10"] / moments2["m00"]), int(moments2["m01"] / moments2["m00"]))

    # Calculate translation (shift to align)
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]

    # Create a translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply the translation to the second image
    aligned_image2 = cv2.warpAffine(image2, translation_matrix, (image2.shape[1], image2.shape[0]))

    # Dynamically resize the stitched image to fit both images
    stitched_width = max(image1.shape[1], dx + image2.shape[1])
    stitched_height = max(image1.shape[0], dy + image2.shape[0])

    # Create an empty canvas to hold the stitched result
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Place the first image at its original position
    stitched_image[:image1.shape[0], :image1.shape[1]] = image1

    # Ensure the dx, dy values are within valid bounds before placing the second image
    if dy >= 0 and dy + image2.shape[0] <= stitched_image.shape[0] and dx >= 0 and dx + image2.shape[1] <= stitched_image.shape[1]:
        # Place the second image in the translated position
        stitched_image[dy:dy + image2.shape[0], dx:dx + image2.shape[1]] = aligned_image2
    else:
        print("The second image goes out of bounds, adjustment required.")

    
else:
    print("Shapes are not similar enough to align.")


# Create a plot with multiple subplots to show images
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(cv2.cvtColor(image1_with_contours, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Contours Image 1')
axes[0, 1].imshow(cv2.cvtColor(image2_with_contours, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Contours Image 2')
axes[1, 0].imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Stitched image')

# If you have an aligned image or final result, display that too
# axes[1, 1].imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
# axes[1, 1].set_title('Aligned Image')

# Display the shape similarity in the last plot
axes[1, 1].text(0.5, 0.5, f'Shape similarity: {shape_similarity:.4f}', ha='center', va='center', fontsize=15)
axes[1, 2].axis('off')

# Hide axes for all subplots
for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()