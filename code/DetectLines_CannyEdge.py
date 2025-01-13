import cv2
import numpy as np

def process_frame_for_lines(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detector to detect edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    # If lines are detected
    if lines is not None:
        merged_lines = merge_similar_lines(lines)
        for line in merged_lines:
            x1, y1, x2, y2 = line
            # Draw the merged lines on the frame
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame, merged_lines
    
    return frame, lines

def merge_similar_lines(lines, slope_threshold=0.3, distance_threshold=500):
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


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def find_active_line(lines, idealX, idealY):
    closesToPoint = 10000
    index = None

    for lineIndex, line in enumerate(lines):
        x1, y1, x2, y2 = line
        if calculate_distance(idealX, idealY, x1, y1) < closesToPoint:
            index = lineIndex
            closesToPoint = calculate_distance(idealX, idealY, x1, y1)
        
        if calculate_distance(idealX, idealY, x2, y2) < closesToPoint:
            index = lineIndex
            closesToPoint = calculate_distance(idealX, idealY, x1, y1)
                
    return index

# Open the video
cap = cv2.VideoCapture('Test1Cropped.mp4')

# Create a named window
cv2.namedWindow('Lines Detection', cv2.WINDOW_NORMAL)

# Move the window to the desired position (x, y)
# Adjust these values based on your screen resolution and where you want the window to appear
window_x = 100  # Horizontal position
window_y = 100  # Vertical position
cv2.moveWindow('Lines Detection', window_x, window_y)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame to detect and merge lines
    processed_frame, lines = process_frame_for_lines(frame)

    frame_height, frame_width = frame.shape[:2]  # Get the height and width of the frame


    index = find_active_line(lines, frame_height, frame_width/ 2)

    if index is not None:
        x1, y1, x2, y2 = lines[index]
        cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display the frame with detected and merged lines
    cv2.imshow('Lines Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()