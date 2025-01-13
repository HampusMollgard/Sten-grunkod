import cv2 as cv
import numpy as np

# Video file path or camera index (0 for the default camera)
video_path = 'film3.mp4'  # Replace with your video file path
cap = cv.VideoCapture(video_path)

assert cap.isOpened(), "Failed to open video file. Check the file path or camera connection."

# Parameters for processing
padding = 50  # Padding size
threshold = 50  # Threshold for "almost black" pixels

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    # Original dimensions of the current frame
    original_height, original_width = frame.shape[:2]

    frame = cv.GaussianBlur(frame, (21, 21), 0)
    # Add white padding around the frame
    frame_padded = cv.copyMakeBorder(frame, padding, padding, padding, padding, cv.BORDER_CONSTANT, value=(255, 255, 255))

    # Create a mask to detect almost black pixels
    mask = cv.inRange(frame_padded, (0, 0, 0), (threshold, threshold, threshold))

    # Invert the mask to make non-black pixels white
    img_white_background = cv.bitwise_not(mask)
    img_white_background = cv.cvtColor(img_white_background, cv.COLOR_GRAY2BGR)

    # Convert the padded image to grayscale
    gray = cv.cvtColor(img_white_background, cv.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv.Canny(gray, 50, 150)

    # Find contours in the padded image
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw all contours on the padded frame
    frame_with_contours = frame_padded.copy()
    cv.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)  # Green contours

    # Assume the largest contour corresponds to the "L" shape
    if len(contours) > 0:
        cnt = max(contours, key=cv.contourArea)

        # Approximate the contour to reduce noise
        epsilon = 0.02 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # Draw the approximated contour and corners
        for point in approx:
            x, y = point[0]
            if padding + 15 < x < padding + original_width - 15 and padding + 50 < y < padding + original_height - 15:
                cv.circle(frame_with_contours, (x, y), 15, (0, 0, 255), -1)  # Draw red circles for corners

    # Display the processed frames
    cv.imshow('Canny Edges (Padded)', edges)
    cv.imshow('Contours and Corners (Padded)', frame_with_contours)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()