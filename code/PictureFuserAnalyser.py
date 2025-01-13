import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def compute_slope_and_intercept(x1, y1, x2, y2):
    if x2 == x1:  # Vertical line
        slope = float('inf')
        intercept = x1  # Use x-intercept for vertical lines
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    return slope, intercept

def merge_lines(lines, angle_threshold=10, distance_threshold=20):
    merged_lines = []
    lines = list(lines)  # Convert NumPy array to a list for mutability

    for i, line1 in enumerate(lines):
        if line1 is None:
            continue
        x1, y1, x2, y2 = line1[0]
        slope1, intercept1 = compute_slope_and_intercept(x1, y1, x2, y2)

        for j, line2 in enumerate(lines[i + 1:], start=i + 1):
            if line2 is None:
                continue
            x3, y3, x4, y4 = line2[0]
            slope2, intercept2 = compute_slope_and_intercept(x3, y3, x4, y4)

            # Compare slopes
            angle_diff = abs(math.degrees(math.atan(slope1)) - math.degrees(math.atan(slope2)))
            if angle_diff > angle_threshold:
                continue  # Skip if lines have significantly different angles

            # Check if lines are close (distance between endpoints)
            dist1 = math.hypot(x3 - x2, y3 - y2)
            dist2 = math.hypot(x4 - x1, y4 - y1)
            if dist1 > distance_threshold and dist2 > distance_threshold:
                continue

            # Merge lines (extend endpoints)
            new_x1 = min(x1, x2, x3, x4)
            new_y1 = min(y1, y2, y3, y4)
            new_x2 = max(x1, x2, x3, x4)
            new_y2 = max(y1, y2, y3, y4)
            lines[j] = None  # Mark the second line as merged
            line1[0] = [new_x1, new_y1, new_x2, new_y2]

        merged_lines.append(line1)

    return [line for line in merged_lines if line is not None]

# Load and preprocess the image
img = cv2.imread('img1.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=20)

# Fuse lines
if lines is not None:
    fused_lines = merge_lines(lines)

# Draw the fused lines
img_with_fused_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if fused_lines is not None:
    for line in fused_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_fused_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_with_fused_lines, cv2.COLOR_BGR2RGB))
plt.title('Fused Lines')
plt.axis('off')

plt.tight_layout()
plt.show()