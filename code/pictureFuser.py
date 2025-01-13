import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
img1 = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.png', cv2.IMREAD_GRAYSCALE)

# Step 1: Apply Canny Edge detection
edges1 = cv2.Canny(img1, 50, 150, apertureSize=3)
edges2 = cv2.Canny(img2, 50, 150, apertureSize=3)

# Step 2: Detect lines using Hough Line Transform
lines1 = cv2.HoughLinesP(edges1, 1, np.pi / 120, threshold=50, minLineLength=30, maxLineGap=60)
lines2 = cv2.HoughLinesP(edges2, 1, np.pi / 120, threshold=50, minLineLength=30, maxLineGap=60)

# Step 3: Function to match lines based on angle and distance
def match_lines(lines1, lines2, max_angle_diff=10, max_distance_diff=30):
    matches = []
    
    for line1 in lines1:
        x1, y1, x2, y2 = line1[0]
        angle1 = np.arctan2(y2 - y1, x2 - x1)  # Calculate the angle of the line
        
        for line2 in lines2:
            x1_, y1_, x2_, y2_ = line2[0]
            angle2 = np.arctan2(y2_ - y1_, x2_ - x1_)  # Calculate the angle of the line
            
            # Compare the angles and the distance between lines
            if abs(angle1 - angle2) < np.radians(max_angle_diff):
                dist1 = np.linalg.norm([x1 - x2, y1 - y2])
                dist2 = np.linalg.norm([x1_ - x2_, y1_ - y2_])
                
                if abs(dist1 - dist2) < max_distance_diff:
                    matches.append((line1, line2))
                    
    return matches

# Step 4: Get the matched lines
matches = match_lines(lines1, lines2)

# Step 5: Draw matched lines on both images
img1_with_lines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_with_lines = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

for match in matches:
    line1, line2 = match
    x1, y1, x2, y2 = line1[0]
    x1_, y1_, x2_, y2_ = line2[0]
    
    # Draw lines on img1 (in green)
    cv2.line(img1_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Draw lines on img2 (in green)
    cv2.line(img2_with_lines, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)

# Step 6: Estimate translation based on the matched lines
# We'll use the center of the lines as the "feature" points for translation
translation_x = 0
translation_y = 0
num_matches = len(matches)

for match in matches:
    line1, line2 = match
    x1, y1, x2, y2 = line1[0]
    x1_, y1_, x2_, y2_ = line2[0]
    
    # Calculate the center of the lines
    center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
    center2 = ((x1_ + x2_) // 2, (y1_ + y2_) // 2)
    
    # Estimate translation (average of the differences between matched centers)
    translation_x += (center2[0] - center1[0])
    translation_y += (center2[1] - center1[1])

# Average translation
translation_x /= num_matches
translation_y /= num_matches

# Step 7: Apply translation to the second image using cv2.warpAffine
# Create translation matrix
M = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

# Apply the transformation
height, width = img1.shape
img2_translated = cv2.warpAffine(img2, M, (width, height))

# Step 8: Stitch the images together (simple method)
# Create an image to hold the stitched result
stitched = np.copy(img1)

# Place the translated second image on top of the first one
# (You can improve this by using alpha blending or similar techniques for a smoother result)
stitched[0:height, 0:width] = np.maximum(stitched[0:height, 0:width], img2_translated)

# Step 9: Display the result
plt.figure(figsize=(12, 6))

# Plot Image 1 with Lines
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img1_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Image 1 with Matched Lines')

# Plot Image 2 with Lines
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(img2_with_lines, cv2.COLOR_BGR2RGB))
plt.title('Image 2 with Matched Lines')

# Plot Stitched Image
plt.subplot(1, 3, 3)
plt.imshow(stitched, cmap='gray')
plt.title('Stitched Image')

plt.tight_layout()
plt.show()