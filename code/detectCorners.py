import cv2

# Path to the video file
video_path = "film2.mp4"  # Replace with the path to your video file

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

# Initialize the ORB detector with adjusted parameters
orb = cv2.ORB_create(
    nfeatures=50,               # Number of keypoints to detect
    scaleFactor=1.2,             # Pyramid decimation ratio
    nlevels=8,                   # Number of levels in the pyramid
    edgeThreshold=30,            # Size of the border where features are not detected
    firstLevel=0,                # Level of the pyramid to start
    WTA_K=4,                     # Number of points for orientation
    scoreType=cv2.ORB_HARRIS_SCORE,  # Use Harris corner detection score
    patchSize=10,                # Patch size for descriptor computation
    fastThreshold=40             # Threshold for FAST corner detection
)

while True:
    # Read the next frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints using ORB
    keypoints = orb.detect(gray, None)

    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(
        frame, keypoints, None, color=(0, 255, 0), flags=cv2.DrawMatchesFlags_DEFAULT
    )

    # Display the frame with keypoints
    cv2.imshow("ORB Keypoints", frame_with_keypoints)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()