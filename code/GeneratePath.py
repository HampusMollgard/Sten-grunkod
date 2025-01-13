import cv2
import numpy as np
from ultralytics import YOLO
import time


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
        return merged_lines
    
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

def locate_green_spaces(frame):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([50, 100, 40])  # Adjust these values as needed
    upper_green = np.array([100, 255, 255])

    # Create a mask for the green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours if needed
            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                points.append((cX, cY))
    return mask, points

def ccw(A, B, C):
    """Funktion som hjälper till att avgöra om tre punkter är i moturs ordning"""
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def lines_intersect(A, B, C, D):
    """Kontrollerar om linjesegmenten AB och CD skär varandra"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def line_intersection(A, B, C, D):
    """Räknar ut skärningspunkten mellan linjesegmenten AB och CD"""
    # Line AB represented as a1x + b1y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1 * A[0] + b1 * A[1]
    
    # Line CD represented as a2x + b2y = c2
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2 * C[0] + b2 * C[1]
    
    determinant = a1 * b2 - a2 * b1
    
    if determinant == 0:
        return None  # Linjerna är parallella
    
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return (int(x), int(y))

def check_line_square_intersection(line, square):
    # Linjen är definierad av (x1, y1) och (x2, y2)
    A = (line[0], line[1])
    B = (line[2], line[3])
    
    # Fyrkanten är definierad av (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    top_left = (square[0], square[1])
    top_right = (square[2], square[1])
    bottom_right = (square[2], square[3])
    bottom_left = (square[0], square[3])
    
    # Kolla om linjen skär någon av fyrkantens sidor
    edges = [(top_left, top_right), (top_right, bottom_right), (bottom_right, bottom_left), (bottom_left, top_left)]
    
    intersections = []
    
    for edge in edges:
        if lines_intersect(A, B, edge[0], edge[1]):
            intersection = line_intersection(A, B, edge[0], edge[1])
            if intersection:
                intersections.append(intersection)
    
    return intersections

def point_inside_box(x, y, box):
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max


# Open the video
cap = cv2.VideoCapture('ManyPictures.mp4')

# Load the YOLOv8 model
model = YOLO('Best.pt')

# Create a named window
cv2.namedWindow('Lines Detection', cv2.WINDOW_NORMAL)

# Move the window to the desired position (x, y)
# Adjust these values based on your screen resolution and where you want the window to appear
window_x = 100  # Horizontal position
window_y = 100  # Vertical position
cv2.moveWindow('Lines Detection', window_x, window_y)


while True:
    ret, frame = cap.read()
    frame  = cv2.resize(frame, (800, 800))
    if not ret:
        break

    processed_frame = frame
    # Process the frame to detect, merge and classify lines
    lines = process_frame_for_lines(frame)
    frame_height, frame_width = frame.shape[:2]  # Get the height and width of the frame
    index = find_active_line(lines, frame_height, frame_width/ 2)

    #Run inference and extract important information from results
    # Locate green spaces in the frame
    mask, green_points = locate_green_spaces(frame)
    for point in green_points:
        cv2.circle(processed_frame, (point[0], point[1]), 5, (255, 0, 0), -1)  # Draw center point in blue
    new_lines = []            
    if green_points is not None:
        results = model(frame, conf=0.50, iou=0.5)  # Set confidence and IoU thresholds
        detections = results[0].boxes
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)
            confidence = box.conf[0]       # Confidence score
            class_id = int(box.cls[0])     # Class ID (index of the detected object)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Calculate midpoints of the bounding box edges
            top_mid = (center_x, y1)       # Midpoint of top edge
            bottom_mid = (center_x, y2)    # Midpoint of bottom edge
            left_mid = (x1, center_y)      # Midpoint of left edge
            right_mid = (x2, center_y)     # Midpoint of right edge

            # List of midpoints
            edge_midpoints = [top_mid, bottom_mid, left_mid, right_mid]

            # Determine which sides to keep based on green points relative to bounding box center
            keep_sides = set()  # Will contain 'bottom', 'left', 'right' depending on green point locations
            for green_point in green_points:
                green_x, green_y = green_point

                # Compare green point with the center of the bounding box
                if green_x < center_x and green_y > center_y:  # Bottom-left relative to the center
                    keep_sides.update(['bottom', 'left'])
                elif green_x > center_x and green_y > center_y:  # Bottom-right relative to the center
                    keep_sides.update(['bottom', 'right'])

            # Special case: if both bottom-left and bottom-right green points exist, only keep bottom
            if len([p for p in green_points if p[1] > center_y]) > 1:
                keep_sides = {'bottom'}

            # Check if there are no green points inside the bounding box
            if len([p for p in green_points if point_inside_box(p[0], p[1], box.xyxy[0])]) == 0:
                # Connect the center of the bottom edge to the center of the top edge
                new_lines.append((bottom_mid[0], bottom_mid[1], top_mid[0], top_mid[1]))
                
                # Keep lines that are connected to both bottom and top edges
                keep_sides = {'bottom', 'top'}

            # After determining which sides to keep, connect the midpoints of the kept edges
            if 'bottom' in keep_sides and 'right' in keep_sides:
                new_lines.append((bottom_mid[0], bottom_mid[1], right_mid[0], right_mid[1]))
            if 'bottom' in keep_sides and 'left' in keep_sides:
                new_lines.append((bottom_mid[0], bottom_mid[1], left_mid[0], left_mid[1]))
            if 'bottom' in keep_sides and len(keep_sides) == 1:  # Only bottom is kept
                box_center = (center_x, center_y)  # Center of the bounding box
                new_lines.append((bottom_mid[0], bottom_mid[1], box_center[0], box_center[1]))

            # Continue with the line detection and splitting
            for lineIndex, line in enumerate(lines):
                x1, y1, x2, y2 = line

                # Get intersection points between the line and the bounding box
                intersections = check_line_square_intersection(line, box.xyxy[0])

                # Sort intersections by their distance from the start of the line (x1, y1)
                intersections = sorted(intersections, key=lambda point: calculate_distance(x1, y1, point[0], point[1]))

                # If there are no intersections, continue to the next line
                if len(intersections) == 0:
                    continue

                # Snap the intersections to the nearest midpoint
                snapped_intersections = []
                for intersection in intersections:
                    # Find the nearest midpoint
                    nearest_mid = min(edge_midpoints, key=lambda mid: calculate_distance(intersection[0], intersection[1], mid[0], mid[1]))
                    snapped_intersections.append(nearest_mid)

                # Determine if the line is connected to the sides we want to keep
                keep_line = False
                for intersection in snapped_intersections:
                    if 'bottom' in keep_sides and intersection == bottom_mid:
                        keep_line = True
                    if 'left' in keep_sides and intersection == left_mid:
                        keep_line = True
                    if 'right' in keep_sides and intersection == right_mid:
                        keep_line = True

                # If the line is connected to the right sides, keep it
                if keep_line:
                    # Case: if the first point of the line is outside the box, keep it
                    if not point_inside_box(x1, y1, box.xyxy[0]):
                        new_lines.append((x1, y1, snapped_intersections[0][0], snapped_intersections[0][1]))

                    # Finally, if the last point of the line is outside the box, keep it as well
                    if not point_inside_box(x2, y2, box.xyxy[0]):
                        new_lines.append((snapped_intersections[-1][0], snapped_intersections[-1][1], x2, y2))
    
            
    #Draw the new lines
    for new_line in new_lines:
        cv2.line(processed_frame, (int(new_line[0]), int(new_line[1])), (int(new_line[2]), int(new_line[3])), (0, 255, 0), 2)
    
    # Draw circles at each snapped intersection point
    for point in snapped_intersections:
        cv2.circle(processed_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)  # Red dot for each snapped point


    '''
    if index is not None:
        x1, y1, x2, y2 = lines[index]
        cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''

    # Visualize the results on the frame
    processed_frame = results[0].plot()  # Annotates the detected objects

    # Display the frame with detected and merged lines
    cv2.imshow('Lines Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()