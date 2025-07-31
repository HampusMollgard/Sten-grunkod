import cv2

cap = cv2.VideoCapture("test5.h264")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output4.mp4", fourcc, 30.0, (640, 640))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()