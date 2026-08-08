import cv2
import mediapipe as mp
import os
import csv
import sys

# --- CONFIGURATION ---
DATASET_PATH = r"D:\SIGNLANGUAGEPROJECT\backend_python\archive\Gesture Image Data" 
CSV_FILE = "landmark_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

print(f"🚀 Starting RELATIVE conversion of {DATASET_PATH}...")

# 1. Setup CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"p{i}_{c}" for i in range(21) for c in ['x', 'y']] + ["label"]
        writer.writerow(header)

if not os.path.exists(DATASET_PATH):
    print("❌ Dataset path not found!")
    sys.exit(1)

count = 0
for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)
    if os.path.isdir(folder_path):
        print(f"   Processing: {label}...")
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            if image is None: continue
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    
                    # Get Wrist (Point 0)
                    wrist_x = landmarks[0].x
                    wrist_y = landmarks[0].y
                    
                    data_row = []
                    for lm in landmarks:
                        relative_x = lm.x - wrist_x
                        relative_y = lm.y - wrist_y
                        data_row.extend([relative_x, relative_y])
                    
                    data_row.append(label)

                    with open(CSV_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data_row)
                    count += 1

print(f"✅ DONE! Converted {count} RELATIVE skeletons.")