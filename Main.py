import cv2
import torch
import numpy as np
from sort import Sort  # SORT: Simple Online and Realtime Tracking
from pathlib import Path

# Load the YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can choose yolov5s, yolov5m, etc.

# Initialize SORT tracker
tracker = Sort()

# Define function to process detections
def process_detections(detections, frame):
    # Create an empty list for tracking bounding boxes
    bbox_xywh = []
    confs = []
    
    for *box, conf, cls in detections:  # Unpack detections
        x_center, y_center, width, height = box  # Get the bounding box
        bbox_xywh.append([x_center, y_center, width, height])
        confs.append([conf])
    
    return np.array(bbox_xywh), np.array(confs)

# Load video or capture live video
video_path = "path_to_your_video.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform YOLO detection
    results = model(frame)
    
    # Extract detections (convert to format needed for SORT)
    detections = results.xywh[0]  # xywh format: (center_x, center_y, width, height)
    bbox_xywh, confs = process_detections(detections, frame)
    
    # Use SORT for tracking
    tracked_objects = tracker.update(np.concatenate((bbox_xywh, confs), axis=1))  # update tracker
    
    # Loop over tracked objects and draw boxes
    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj  # obj contains (x1, y1, x2, y2, obj_id)
        
        # Draw bounding box and the object ID on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Object Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()














