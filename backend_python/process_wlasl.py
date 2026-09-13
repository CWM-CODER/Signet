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
    Extracts 3D pose and hand landmarks.

    Pose:
        33 landmarks × 3 coordinates = 99 features

    Left hand:
        21 landmarks × 3 coordinates = 63 features

    Right hand:
        21 landmarks × 3 coordinates = 63 features

    Total:
        99 + 63 + 63 = 225 features per frame
    """

    pose = np.zeros((33, 3), dtype=np.float32)
    lh = np.zeros((21, 3), dtype=np.float32)
    rh = np.zeros((21, 3), dtype=np.float32)

    pose_detected = False
    left_hand_detected = False
    right_hand_detected = False

    # Extract 3D pose landmarks
    if results.pose_landmarks:
        pose_detected = True

        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i] = [lm.x, lm.y, lm.z]

    # Extract 3D left-hand landmarks
    if results.left_hand_landmarks:
        left_hand_detected = True

        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i] = [lm.x, lm.y, lm.z]

    # Extract 3D right-hand landmarks
    if results.right_hand_landmarks:
        right_hand_detected = True

        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i] = [lm.x, lm.y, lm.z]

    # Normalize relative to shoulder center and shoulder width
    if pose_detected:
        left_shoulder = pose[11]
        right_shoulder = pose[12]

        shoulder_center = (left_shoulder + right_shoulder) / 2
        shoulder_width = np.linalg.norm(
            left_shoulder - right_shoulder
        )

        if shoulder_width < 1e-6:
            shoulder_width = 1.0

        pose = (pose - shoulder_center) / shoulder_width

        if left_hand_detected:
            lh = (lh - shoulder_center) / shoulder_width

        if right_hand_detected:
            rh = (rh - shoulder_center) / shoulder_width

    features = np.concatenate([
        pose.flatten(),
        lh.flatten(),
        rh.flatten()
    ])

    return features

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
        frames = np.array(frames, dtype=np.float32)

        if len(frames) < SEQUENCE_LENGTH:
            pad = np.zeros(
                (SEQUENCE_LENGTH - len(frames), 225),
                dtype=np.float32
            )
            frames = np.concatenate([frames, pad])

        elif len(frames) > SEQUENCE_LENGTH:
            frames = frames[:SEQUENCE_LENGTH]

        assert frames.shape == (30, 225), (
            f"Unexpected sequence shape: {frames.shape}"
        )
            
        np.save(save_path, frames)
        processed_count += 1
        print(f"✅ {word_label}: {video_id} | Total: {processed_count}")

print(f"🎉 Done! Data saved to: {OUTPUT_PATH}")
