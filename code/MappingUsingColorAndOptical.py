import cv2
import numpy as np

# Open the video file
video_path = 'LiveRecording.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame to get the size of the frame
ret, frame = cap.read()
if not ret:
    print("Error: Couldn't read the first frame.")
    exit()

frame_height, frame_width, _ = frame.shape

# You can decide how many frames to stitch together horizontally and vertically
# For example, stitch 4 frames horizontally and 3 frames vertically
rows = 3
cols = 4

# Create a blank canvas to place the frames onto
stitched_image = np.zeros((frame_height * rows, frame_width * cols, 3), dtype=np.uint8)

# Reset the video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Loop through the frames and place them onto the stitched image
frame_count = 0
for row in range(rows):
    for col in range(cols):
        ret, frame = cap.read()
        if not ret:
            break
        # Place the frame at the correct position on the stitched image
        stitched_image[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width] = frame
        frame_count += 1

# Release the video capture object
cap.release()

# Show the stitched image
cv2.imshow("Stitched Video Frames", stitched_image)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()