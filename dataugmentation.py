import cv2
import os


supported_formats = [".avi", ".mp4", ".mkv"]


source_folder = '/home/ical/openpose/run'  # 


for filename in os.listdir(source_folder):
    
    if any(filename.endswith(ext) for ext in supported_formats):
        video_file = os.path.join(source_folder, filename)

        
        cap = cv2.VideoCapture(video_file)

        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        output_file = os.path.join(source_folder, f"flipped_{filename}")

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

           
            flipped_frame = cv2.flip(frame, 1)

            
            out.write(flipped_frame)

        
        cap.release()
        out.release()

print("Finish")
