from ultralytics import YOLO
import yolo_module 
from openpose_module import initialize_openpose
import openpose_module 
import tensorflow as tf
from keras.models import load_model
from action_prediction import run_frame_loop
# 初始化
opWrapper = initialize_openpose()
modelFile = "/home/ical/PenguinChuan/openpose/0629.h5"
model = load_model(modelFile, compile=None)
# 影片循環處理
def process_video(video_path, output_path):
    run_frame_loop(video_path, yolo_model, opWrapper, output_path)
if __name__ == "__main__":
    labels = ["TEAM0", "TEAM1", "GK-L", "GK-R", "Ball", "Main Ref", "Side Ref", "Staff"]
    box_colors = {
        "0": (193, 82, 17),#藍色
        "1": (255, 255, 255),#白色
        "2": (41, 248, 165),# 綠色
        "3": (41, 248, 165),# 綠色
        "4": (155, 62, 157),# 紫色
        "5": (23, 13, 227),#紅色
        "6": (23, 13, 227),#紅色
        "7": (22, 11, 15)#黑色
    }
    video_path = "/home/ical/PenguinChuan/openpose/Birdseyeview/inference/output/task5_output.mp4"
    output_path = '/home/ical/PenguinChuan/openpose/Football-Object-Detection-main/output/task5output0703.mp4'
    yolo_model = YOLO("/home/ical/PenguinChuan/openpose/Football-Object-Detection-main/weights/best.pt")
    process_video(video_path, yolo_model, output_path)
