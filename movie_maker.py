from ultralytics import YOLO
import cv2 
from matplotlib import pyplot as plt
import torch
import numpy as np

model = YOLO("trained models/yolo11s_trained.pt" , task = "detect", verbose = True,)
video_path= r'data\Project_20250719_085238_1.mp4'.replace('\\','/')
output_video_path='output_videos'

# Open the video file
cap = cv2.VideoCapture(video_path)
# Get video properties for output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4


 # Create VideoWriter object
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame) # Adjust confidence threshold

    # Annotate the frame with predictions
    annotated_frame = results[0].plot()

    # Display the annotated frame
    #cv2.imshow("YOLO Predictions", annotated_frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    
    # Write the annotated frame to the output video
    out.write(annotated_frame)
    # Exit on 'q' key press
    

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
