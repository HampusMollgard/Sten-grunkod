import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.pt')

# Path to the video file
video_path = 'Test1Cropped.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

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

# Loop over frames
while True:
    # Read frame by frame
    ret, frame = cap.read()
    
    # If the frame was not read successfully, break the loop
    if not ret:
        break
    
    # Use the YOLO model to detect objects on the current frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Annotates the detected objects
    
    detections = results[0].boxes

    # Locate green spaces in the frame
    mask, green_points = locate_green_spaces(frame)

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates (x1, y1, x2, y2)
        confidence = box.conf[0]       # Confidence score
        class_id = int(box.cls[0])     # Class ID (index of the detected object)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        x1Line = int((x1 + x2) / 2)
        y1Line = int(y2)
        x2Line = x1Line
        y2Line = int(y1)
        #Only takes the first green point into consideration, needs to be fixed
        if green_points[0][1] > center_y:
            y2Line = center_y
            if green_points[0][0] > center_x:
                x2Line = int(x2)
            else:
                x2Line = int(x1)

        cv2.line(annotated_frame, (x1Line, y1Line), (x2Line, y2Line), (255, 0, 0), 2)
        
                    


    for point in green_points:
        cv2.circle(annotated_frame, (point[0], point[1]), 5, (255, 0, 0), -1)  # Draw center point in blue
    print(green_points)

    # Display the frame with the annotations
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    
    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()