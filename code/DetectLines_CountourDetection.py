import cv2
import numpy as np

def process_frame_with_morphology(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny Edge Detector to detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define a kernel for morphological operations
    kernel = np.ones((50, 50), np.uint8)
    
    # Apply morphological closing to merge nearby edges (dilation followed by erosion)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=10)


    # Draw detected lines on the original frame
    if lines is not None:
        merged_lines = merge_similar_lines(lines)
        for line in merged_lines:
            x1, y1, x2, y2 = line
            # Draw the merged lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame


def merge_similar_lines(lines, slope_threshold=0.3, distance_threshold=50):
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        new_line = True
        
        # Calculate slope of the current line
        slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
        
        for merged_line in merged_lines:
            mx1, my1, mx2, my2 = merged_line
            merged_slope = (my2 - my1) / (mx2 - mx1 + 1e-6)
            
            # If lines are parallel (similar slope) and close to each other, merge them
            if abs(slope - merged_slope) < slope_threshold:
                if np.linalg.norm(np.array([x1, y1]) - np.array([mx1, my1])) < distance_threshold:
                    # Merge the lines by averaging the coordinates
                    mx1 = (x1 + mx1) // 2
                    my1 = (y1 + my1) // 2
                    mx2 = (x2 + mx2) // 2
                    my2 = (y2 + my2) // 2
                    merged_line[:] = [mx1, my1, mx2, my2]
                    new_line = False
                    break
        
        if new_line:
            merged_lines.append([x1, y1, x2, y2])
    
    return merged_lines

# Open the video file (replace with 0 for webcam input)
cap = cv2.VideoCapture('Test1Cropped.mp4')

while True:
    # Read the current frame from the video
    ret, frame = cap.read()
    
    # If there are no more frames, break the loop
    if not ret:
        break
    
    # Process the frame to detect lines using morphological operations
    processed_frame = process_frame_with_morphology(frame)
    
    # Display the processed frame with detected lines
    cv2.imshow('Morphological Line Detection', processed_frame)
    
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()