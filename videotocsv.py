from distutils.errors import PreprocessError
import cv2
import numpy as np
import keras
from keras.models import load_model
import sys
import csv
import os
sys.path.append('/home/ical/openpose/ultralytics')
from ultralytics import YOLO
from collections import defaultdict

sys.path.append('/home/ical/openpose/build/python')
from openpose import pyopenpose as op

# YOLOv8 model
model = YOLO('yolov8n.pt') 
class_names = ['person']

n_steps = 16  
iterator = 0
openpose_output = []  # Will store the openpose time series data for recent n_steps
interval = 1
sequence_start = 0  # starting location of circular array

# OpenPose params
params = dict()
params["model_folder"] = "/home/ical/openpose/models"
params["model_pose"] = "BODY_25"

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Specify the folder containing the videos
videos_folder = "/home/ical/openpose/0908NUK/walk"

# Open CSV file for writing
csv_filename = '/home/ical/openpose/0908NUK/walk.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Video_Name', 'Person_ID', 'Keypoints'])


supported_formats = [".avi", ".mp4", ".mkv"]

# Iterate over all video files in the folder
for video_file in os.listdir(videos_folder):
    if any(video_file.endswith(ext) for ext in supported_formats):
        video_path = os.path.join(videos_folder, video_file)

        # Extract video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        track_history = defaultdict(lambda: []) 
        frame_counter = 0
        missing_frames_counter = 0  

        while cap.isOpened() and frame_counter < 16:
            success, frame = cap.read()
            if success:
                target_size = (480, 720)
                frame = cv2.resize(frame, target_size)

                datum = op.Datum()
                datum.cvInputData = frame
                opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                frame = datum.cvOutputData
                results = model.track(frame, classes=0, persist=True)
                boxes = results[0].boxes.xywh.cpu()
                keypoints_list = datum.poseKeypoints

                if keypoints_list is None or not any(keypoints.any() for keypoints in keypoints_list):
                    missing_frames_counter += 1
                    if missing_frames_counter >= 20:
                        print(f"No keypoints detected for 20 frames in {video_name}, skipping to next video.")
                        break  
                    continue  
                else:
                    missing_frames_counter = 0 

                for person_id, keypoints in enumerate(keypoints_list):
                    if keypoints.any():
                        keypoints = keypoints[:, :-1]
                        row = [video_name, person_id]
                        for i, keypoint in enumerate(keypoints):
                            x = keypoint[0]
                            y = keypoint[1]
                            row.insert(2 * i + 2, x)
                            row.insert(2 * i + 3, y)
                        csv_writer.writerow(row)

                frame_counter += 1  # Increment frame counter
            else:
                break  # Break the loop when the video ends

cap.release()
cv2.destroyAllWindows()
csv_file.close()
