import os
import shutil
import random

def split_data(source_folder, test_folder, test_ratio=0.2):
    
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    video_files = [f for f in all_files if any(f.endswith(ext) for ext in supported_formats)]
    
    
    test_size = int(len(video_files) * test_ratio)
    
   
    test_files = random.sample(video_files, test_size)
    
   
    os.makedirs(test_folder, exist_ok=True)
    
   
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))

    print(f"Moved {test_size} files to {test_folder}")

supported_formats = [".avi", ".mp4", ".mkv"]


base_folder = '/home/ical/NUK'  
#subfolders = ['run', 'stand', 'kick']
subfolders = ['kick']
test_suffix = '_test'

for subfolder in subfolders:
    source_folder = os.path.join(base_folder, subfolder)
    test_folder = os.path.join(base_folder, subfolder + test_suffix)
    split_data(source_folder, test_folder)

print("Finish")
