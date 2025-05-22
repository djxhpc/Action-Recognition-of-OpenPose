import numpy as np
from keras.models import load_model
import tensorflow as tf
from collections import defaultdict
import cv2
import time
from collections import Counter
from collections import deque
from distutils.errors import PreprocessError
from scipy.interpolate import interp1d
from openpose_module import get_keypoints_from_frame
from openpose import pyopenpose as op
from yolo_module import detect_ball, get_grass_color, get_players_boxes, get_kits_colors,get_grass_color, classify_kits, get_kits_classifier, get_left_team_label, crop_and_zoom

labels = ["TEAM0", "TEAM1", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
box_colors = {
        "0": (193, 82, 17),#藍色
        "1": (255, 255, 255),#白色
        "2": (41, 248, 165),#綠色
        "3": (41, 248, 165),#綠色
        "4": (155, 62, 157),#紫色
        "5": (23, 13, 227),#紅色
        "6": (23, 13, 227),#紅色
        "7": (22, 11, 15)#黑色
    }

LABELS = ["RUNNING", "STANDING", "WALKING","STRONG KICKING","PASSING"]
modelFile = "/home/ical/PenguinChuan/openpose/0629.h5"
model = load_model(modelFile, compile=None)
opWrapper = op.WrapperPython()
track_history = defaultdict(lambda: [])
n_steps = 16 
# 限制 GPU 記憶體使用
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
            )
    except RuntimeError as e:
        print(e)

def predict_behavior(keypoints_series):
    X = np.array([keypoints_series])
    X = X[:, :, :, :2]
    X = X.reshape(X.shape[0], X.shape[1], -1)
    y_pred = model.predict(X)
    predicted_label = np.argmax(y_pred, axis=1)[0]
    return LABELS[predicted_label]

def run_frame_loop(video_path, yolo_model, opWrapper, action_model, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    frame_count = 0
    behavior_counter = Counter()
    stats_text = ""
    start_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        result = yolo_model(frame, conf=0.5, verbose=False)[0]
        keypoints_list, output_data = get_keypoints_from_frame(opWrapper, frame)

        if keypoints_list is not None and len(keypoints_list) > 0:
            for person_id, keypoints in enumerate(keypoints_list):
                if keypoints.any():
                    if frame_count % n_steps == 0:
                        track_history[person_id].append(keypoints)
                        if len(track_history[person_id]) > n_steps:
                            track_history[person_id] = track_history[person_id][-n_steps:]
                            if len(track_history[person_id]) == n_steps:
                                predicted_behavior = predict_behavior(track_history[person_id])
                                nose = keypoints[0]
                                x, y = int(nose[0]), int(nose[1])
                                cv2.putText(output_data, predicted_behavior, (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                behavior_counter[predicted_behavior] += 1

        for box in result.boxes:
            label = int(box.cls.cpu().numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            

        if frame_count % 16 == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                start_time = time.time()
                total_behaviors = sum(behavior_counter.values())
                if total_behaviors > 0:
                    stats_text = "Behavior Percentages: "
                    for behavior, count in behavior_counter.items():
                        percentage = (count / total_behaviors) * 100
                        stats_text += f"{behavior}: {percentage:.2f}% "
                else:
                    stats_text = "No behaviors detected"
                behavior_counter.clear()

        cv2.putText(frame, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        output_video.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
