import cv2
import numpy as np

# Load video
video_path = 'test.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Read rotation data from a file
with open("test", "r") as file:
    data = file.readlines()

data_line = 0

# Verify if the video is loaded
if not cap.isOpened():
    print("Could not open the video!")
    exit()

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Could not read the first frame!")
    cap.release()
    exit()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create a large blank canvas for the map
canvas_size = (2000, 2000)  # Adjust this size based on your expected mapping area
canvas = np.ones(canvas_size, dtype=np.uint8) * 255  # White canvas
canvas_center = (canvas_size[1] // 2, canvas_size[0] // 2)  # Center of the canvas

# Initialize the ORB detector
orb = cv2.ORB_create()

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

rotation_angle = 0  # Initial rotation angle in degrees
frame_counter = 0  # Initialize frame counter
blending_factor = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    frame_counter += 1  # Increment the frame counter

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for better keypoint detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if len(keypoints) >= 2:
        # Match features using optical flow
        prev_points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

        # Calculate optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)

        if next_points is not None and len(next_points) >= 3:
            good_old = prev_points[status.ravel() == 1]
            good_new = next_points[status.ravel() == 1]

            # Update rotation angle from data
            if data_line < len(data) - 3:
                rotation_angle = float(data[data_line]) - float(data[data_line - 3])
                rotation_angle = -rotation_angle
                data_line += 3

            try:
                # Estimate affine transformation
                affine_matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                if affine_matrix is not None:
                    dx, dy = affine_matrix[0, 2], affine_matrix[1, 2]

                    # Translation matrix
                    translation_matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
                    # Rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(canvas_center, rotation_angle, 1)

                    # Apply transformations
                    canvas = cv2.warpAffine(canvas, translation_matrix, canvas_size, flags=cv2.INTER_LINEAR, borderValue=255)
                    canvas = cv2.warpAffine(canvas, rotation_matrix, canvas_size, flags=cv2.INTER_LINEAR, borderValue=255)
            except Exception as e:
                print(f'Error during transformation: {e}')
                continue

    # Overlay the new frame at the canvas center
    h, w = gray.shape
    x1, y1 = canvas_center[0] - w // 2, canvas_center[1] - h // 2
    x2, y2 = x1 + w, y1 + h

    # Ensure bounds within canvas
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, canvas_size[1]), min(y2, canvas_size[0])

    # Blend the frame with the canvas
    new_frame_cropped = gray[:y2 - y1, :x2 - x1]
    canvas_cropped = canvas[y1:y2, x1:x2]

    black_pixel_mask = new_frame_cropped < 50  # Mask for black pixels
    blended = canvas_cropped.astype(np.float32)
    blended[black_pixel_mask] = (
        (1 - blending_factor) * canvas_cropped[black_pixel_mask].astype(np.float32) +
        blending_factor * new_frame_cropped[black_pixel_mask].astype(np.float32)
    )
    canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    # Display the canvas
    cv2.imshow('Mapping Canvas', canvas)

    # Exit on pressing 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()  # Update the previous frame

# Save the final canvas
cv2.imwrite("mapped_path.png", canvas)

# Release resources
cap.release()
cv2.destroyAllWindows()