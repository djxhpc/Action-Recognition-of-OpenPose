import cv2
import time
from ultralytics import YOLO

# Load the YOLO model

# model = YOLO('/home/ical/openpose/market/runs/detect/train11/weights/best.pt')#yolov8n
# model = YOLO('/home/ical/openpose/market/runs/detect/train12/weights/best.pt')#yolov8s
model = YOLO('/home/ical/openpose/market/runs/detect/train13/weights/best.pt')#yolov8m
# Open the video file
video_path = '/home/ical/openpose/0611.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
total_inference_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    start_time = time.time()
    results = model(frame)
    end_time = time.time()

    # Calculate inference time
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    total_inference_time += inference_time
    frame_count += 1

    # Display the resulting frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Calculate average FPS
average_inference_time = total_inference_time / frame_count
fps = 1 / (average_inference_time / 1000)  # Convert back to seconds

print(f"Average Inference Time: {average_inference_time:.2f} ms")
print(f"FPS: {fps:.2f}")
