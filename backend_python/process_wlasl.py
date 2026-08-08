import cv2
import mediapipe as mp
import numpy as np
import os
import json
import sys

VIDEO_PATH = r"D:/SignLanguageProject/archive/videos"
JSON_PATH = r"D:/SignLanguageProject/archive/WLASL_v0.3.json"

OUTPUT_PATH = r"D:/SignLanguageProject/archive/Word_Training_Data_Normalized"

TARGET_WORDS = [
    "hello", "no", "yes", "walk", "book", 
    "computer", "finish", "late", "many", "year"
]

SEQUENCE_LENGTH = 30 

mp_holistic = mp.solutions.holistic

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

print(f"🚀 Processing {len(TARGET_WORDS)} words with RELATIVE NORMALIZATION...")

if not os.path.exists(JSON_PATH) or not os.path.exists(VIDEO_PATH):
    print("❌ Error: Check your JSON and VIDEO paths!")
    sys.exit(1)

try:
    with open(JSON_PATH, 'r') as f:
        wlasl_data = json.load(f)
except Exception as e:
    print(f"❌ Error reading JSON: {e}")
    sys.exit(1)

id_to_word = {}
found_words_count = 0

for entry in wlasl_data:
    word = entry['gloss']
    if word in TARGET_WORDS:
        found_words_count += 1
        for instance in entry['instances']:
            video_id = instance['video_id']
            id_to_word[video_id] = word

print(f"✅ Found {len(id_to_word)} videos for {found_words_count} words.")

def get_relative_keypoints(results):
    """
    Converts raw screen coordinates to body-relative coordinates.
    1. Finds center of shoulders.
    2. Subtracts center from all points.
    3. Divides by shoulder width to handle distance.
    4. Ignores Z-axis (Webcams are bad at depth).
    """
   
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 2))
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 2))
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 2))

    if results.pose_landmarks:
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # Calculate scale (width of shoulders)
        width = np.linalg.norm(left_shoulder - right_shoulder)
        if width == 0: width = 1.0 # Prevent div/0
        
        pose = (pose - [center_x, center_y]) / width
        
        # Normalize Hands (relative to body center)
        if results.left_hand_landmarks: lh = (lh - [center_x, center_y]) / width
        if results.right_hand_landmarks: rh = (rh - [center_x, center_y]) / width

    # Flatten (33*2 + 21*2 + 21*2 = 150 inputs)
    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

processed_count = 0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    video_files = os.listdir(VIDEO_PATH)
    
    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        if video_id not in id_to_word: continue
            
        word_label = id_to_word[video_id]
        save_folder = os.path.join(OUTPUT_PATH, word_label)
        if not os.path.exists(save_folder): os.makedirs(save_folder)
        
        save_path = os.path.join(save_folder, f"{video_id}.npy")
        if os.path.exists(save_path): continue

        video_full_path = os.path.join(VIDEO_PATH, video_file)
        cap = cv2.VideoCapture(video_full_path)
        
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(get_relative_keypoints(results))
        cap.release()
        
        if len(frames) == 0: continue
        
        # Normalize to 30 frames
        frames = np.array(frames)
        if len(frames) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(frames), frames.shape[1]))
            frames = np.concatenate([frames, pad])
        elif len(frames) > SEQUENCE_LENGTH:
            frames = frames[:SEQUENCE_LENGTH]
            
        np.save(save_path, frames)
        processed_count += 1
        print(f"✅ {word_label}: {video_id} | Total: {processed_count}")

print(f"🎉 Done! Data saved to: {OUTPUT_PATH}")