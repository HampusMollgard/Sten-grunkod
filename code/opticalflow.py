import cv2
import numpy as np

# Load video
video_path = 'LiveRecording.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open the video!")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Could not read the first frame!")
    cap.release()
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Preprocessing: Enhance contrast with adaptive thresholding
prev_thresh = cv2.adaptiveThreshold(prev_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Feature detection settings
feature_params = dict(maxCorners=200, qualityLevel=0.7, minDistance=10, blockSize=10)

# Optical flow settings
lk_params = dict(winSize=(15, 15), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detect initial features
prev_points = cv2.goodFeaturesToTrack(prev_thresh, mask=None, **feature_params)

# Mask for drawing
mask = np.zeros_like(prev_frame)

# Frame counter for refreshing points
frame_counter = 0
refresh_interval = 10  # Refresh points every 10 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Refresh points every `refresh_interval` frames
    if frame_counter % refresh_interval == 0 or prev_points is None:
        print("Refreshing points...")
        prev_points = cv2.goodFeaturesToTrack(thresh, mask=None, **feature_params)

    # Check if there are points to track
    if prev_points is not None:
        # Calculate optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_thresh, thresh, prev_points, None, **lk_params)

        if next_points is not None and status is not None:
            # Filter good points
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]

            # Draw the movement
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                mask = cv2.line(mask, (int(x_new), int(y_new)), (int(x_old), int(y_old)), (0, 255, 0), 2)
                frame = cv2.circle(frame, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)

            # Update previous points
            prev_points = good_new.reshape(-1, 1, 2)
        else:
            print("No good points found, recalculating...")
            prev_points = cv2.goodFeaturesToTrack(thresh, mask=None, **feature_params)

    # Update the previous thresholded frame
    prev_thresh = thresh.copy()

    # Combine the mask and frame
    output = cv2.add(frame, mask)

    # Show the result
    cv2.imshow('Optical Flow', output)

    # Increment the frame counter
    frame_counter += 1

    # Exit on 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()