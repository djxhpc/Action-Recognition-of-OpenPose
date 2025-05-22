import cv2
import numpy as np
from ultralytics import YOLO

def detect_ball(model, frame):
    result = model(frame, conf=0.5, verbose=False)[0]
    for box in result.boxes:
        label = int(box.cls.cpu().numpy()[0])  
        if label == 2:  # Assuming label 2 is for the ball
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  
            return (x1, y1, x2, y2)
    return None

# Helper functions (unchanged)
def get_grass_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    grass_hue = np.argmax(hist[:, 10:246])
    return cv2.cvtColor(np.uint8([[[grass_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]

def get_players_boxes(result):
    players_imgs = []
    players_boxes = []
    for box in result.boxes:
        if int(box.cls.cpu().numpy()[0]) == 0:  # Assuming label 0 is for players
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            players_imgs.append(result.orig_img[y1:y2, x1:x2])
            players_boxes.append(box)
    return players_imgs, players_boxes

def get_kits_colors(players, grass_hsv=None, frame=None):
    kits_colors = []
    if grass_hsv is None:
        grass_color = get_grass_color(frame)
        grass_hsv = cv2.cvtColor(np.uint8([[list(grass_color)]]), cv2.COLOR_BGR2HSV)

    for player_img in players:
        hsv = cv2.cvtColor(player_img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([grass_hsv[0, 0, 0] - 10, 40, 40])
        upper_green = np.array([grass_hsv[0, 0, 0] + 10, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(mask)
        upper_mask = np.zeros(player_img.shape[:2], np.uint8)
        upper_mask[0:player_img.shape[0]//2, 0:player_img.shape[1]] = 255
        mask = cv2.bitwise_and(mask, upper_mask)
        kit_color = np.array(cv2.mean(player_img, mask=mask)[:3])
        kits_colors.append(kit_color)
    return kits_colors

def classify_kits(kits_clf, kit_color):
    kit_color = np.array(kit_color).reshape(1, -1)
    return kits_clf.predict(kit_color)

def get_kits_classifier(kits_colors):
    from sklearn.cluster import KMeans
    if len(kits_colors) < 2:
        return None
    kits_colors = np.array(kits_colors)
    kits_kmeans = KMeans(n_clusters=2, n_init="auto")
    kits_kmeans.fit(kits_colors)
    return kits_kmeans

def get_left_team_label(players_boxes, kits_colors, kits_clf):
    left_team_label = 0
    team_0 = []
    team_1 = []
    for i in range(len(players_boxes)):
        x1, y1, x2, y2 = map(int, players_boxes[i].xyxy[0].cpu().numpy())
        team = classify_kits(kits_clf, [kits_colors[i]]).item()
        if team == 0:
            team_0.append(np.array([x1]))
        else:
            team_1.append(np.array([x1]))
    team_0 = np.array(team_0)
    team_1 = np.array(team_1)
    if np.average(team_0) - np.average(team_1) > 0:
        left_team_label = 1
    return left_team_label

def crop_and_zoom(frame, ball_box, zoom_factor=18, crop_size=100, window_size=(300, 300)):
    x1, y1, x2, y2 = ball_box
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    h, w = frame.shape[:2]
    crop_size = max(x2 - x1, y2 - y1) * zoom_factor
    crop_x1 = max(0, cx - crop_size // 2)
    crop_y1 = max(0, cy - crop_size // 2)
    crop_x2 = min(w, cx + crop_size // 2)
    crop_y2 = min(h, cy + crop_size // 2)
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    zoomed_frame = cv2.resize(cropped_frame, window_size)
    return zoomed_frame
