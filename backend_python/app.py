import os
import cv2
import numpy as np
import base64
import time
import pickle
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from supabase import create_client, Client
from dotenv import load_dotenv
import mediapipe as mp
import sys
import traceback
from collections import deque, Counter

try:
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: No NVIDIA GPU detected!")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Backend Running on GPU: {torch.cuda.get_device_name(0)}")

    print("Loading .env file...")
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL or SUPABASE_KEY is missing from .env file!")

    print("Connecting to Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase Connected.")

    app = Flask(__name__)
    CORS(app)

    class LandmarkNet(nn.Module):
        def __init__(self, num_classes):
            super(LandmarkNet, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(42, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.model(x)

    print("Loading LETTER Model...")
    if not os.path.exists("model.pth") or not os.path.exists("labels.pkl"):
        raise FileNotFoundError("model.pth or labels.pkl missing for Letter Model!")

    with open("labels.pkl", "rb") as f:
        letter_le = pickle.load(f)

    letter_model = LandmarkNet(len(letter_le.classes_)).to(device)
    letter_model.load_state_dict(torch.load("model.pth", map_location=device))
    letter_model.eval()
    print("Letter Model loaded!")

    class LSTMModel(nn.Module):
        def __init__(self, input_size=150, hidden_size=64, num_layers=1, num_classes=10):
            super(LSTMModel, self).__init__()
            self.feature_extract = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(30),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
            self.dropout = nn.Dropout(0.4)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.feature_extract(x)
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.relu(self.fc1(out))
            out = self.fc2(out)
            return out

    print("Loading WORD Model...")
    word_model_path = "models/action_lstm.pth"
    word_labels_path = "models/action_labels.pkl"

    word_model = None
    word_actions = []

    if os.path.exists(word_model_path) and os.path.exists(word_labels_path):
        with open(word_labels_path, "rb") as f:
            word_actions = pickle.load(f)
        word_model = LSTMModel(num_classes=len(word_actions)).to(device)
        word_model.load_state_dict(torch.load(word_model_path, map_location=device))
        word_model.eval()
        print(f"Word Model loaded! ({len(word_actions)} words)")
    else:
        print("WARNING: Word model not found. 'Speaking Mode' will be disabled.")

    mp_hands = mp.solutions.hands
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

except Exception as e:
    print("\n\n!!! STARTUP ERROR !!!")
    print(str(e))
    print(traceback.format_exc())
    sys.exit(1)

word_buffer = []
current_word = ""
last_char = ""
last_char_time = 0
REPEAT_DELAY = 1.5
last_activity_time = time.time()
BUFFER_LIMIT = 10
TIMEOUT_SECONDS = 5.0
WORD_FINISH_DELAY = 2.0

last_sent_message = ""
STABILITY_THRESHOLD = 5
stability_count = 0
stable_char = ""

sequence = []
frame_counter = 0
SKIP_RATE = 4
COOLDOWN_TIME = 2.5
last_prediction_time = 0

THRESHOLD_CONFIDENCE = 0.85

def are_hands_detected(results):
    return (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 2))
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 2))
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 2))

    if results.pose_landmarks:
        center = (pose[11] + pose[12]) / 2
        width = np.linalg.norm(pose[11] - pose[12])
        if width == 0: width = 1.0

        pose = (pose - center) / width
        if results.left_hand_landmarks: lh = (lh - center) / width
        if results.right_hand_landmarks: rh = (rh - center) / width

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten()])

@app.route('/predict', methods=['POST'])
def predict():
    global current_word, last_char, word_buffer, last_activity_time, last_char_time
    global stability_count, stable_char, last_sent_message, sequence
    global frame_counter, last_prediction_time

    try:
        data = request.json
        base64_string = data.get('image')
        mode = data.get('mode', 'letters')

        if "," in base64_string:
            encoded_data = base64_string.split(',')[1]
        else:
            encoded_data = base64_string

        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({"error": "Bad Image"}), 400

        processed_image_b64 = ""
        predicted_text = ""
        final_message = ""

        if mode == 'letters':
            sequence = []

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            predicted_char = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    landmarks = hand_landmarks.landmark
                    wrist_x = landmarks[0].x
                    wrist_y = landmarks[0].y
                    relative_data = []
                    for lm in landmarks: relative_data.extend([lm.x - wrist_x, lm.y - wrist_y])

                    input_tensor = torch.tensor([relative_data], dtype=torch.float32).to(device)

                    with torch.no_grad():
                        outputs = letter_model(input_tensor)
                        _, predicted = torch.max(outputs, 1)
                        raw_char = letter_le.inverse_transform([predicted.item()])[0]

                    if raw_char == stable_char: stability_count += 1
                    else:
                        stability_count = 0
                        stable_char = raw_char

                    if stability_count >= STABILITY_THRESHOLD:
                        predicted_char = stable_char

            current_time = time.time()
            if predicted_char:
                last_activity_time = current_time
                if predicted_char != last_char:
                    last_char = predicted_char
                    last_char_time = current_time
                    if predicted_char == 'del': current_word = current_word[:-1]
                    elif predicted_char == 'nothing': pass
                    elif predicted_char != 'space': current_word += predicted_char
                elif predicted_char == last_char:
                    if (current_time - last_char_time) > REPEAT_DELAY:
                        if predicted_char not in ['space', 'del', 'nothing']:
                            current_word += predicted_char
                            last_char_time = current_time

            if current_word and (current_time - last_activity_time > WORD_FINISH_DELAY):
                word_buffer.append(current_word)
                current_word = ""

            predicted_text = predicted_char

        elif mode == 'words':
            if word_model is None: return jsonify({"error": "Word Model not loaded"}), 500

            current_word = ""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            if not are_hands_detected(results):
                sequence = []
                predicted_text = ""
            else:
                if time.time() - last_prediction_time < COOLDOWN_TIME:
                    predicted_text = "..."
                else:
                    frame_counter += 1
                    if frame_counter % SKIP_RATE == 0:

                        keypoints = extract_keypoints(results)
                        sequence.append(keypoints)
                        sequence = sequence[-30:]

                        if len(sequence) == 30:
                            input_array = np.array([sequence])
                            input_tensor = torch.tensor(input_array, dtype=torch.float32).to(device)

                            with torch.no_grad():
                                res = word_model(input_tensor)
                                probabilities = torch.nn.functional.softmax(res, dim=1)[0]
                                confidence, best_idx = torch.max(probabilities, 0)

                            if confidence.item() > THRESHOLD_CONFIDENCE:
                                word_detected = word_actions[best_idx.item()]

                                word_buffer.append(word_detected)
                                last_activity_time = time.time()
                                predicted_text = word_detected

                                last_prediction_time = time.time()
                                sequence = []

        _, buffer = cv2.imencode('.jpg', frame)
        processed_image_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        should_send = (len(word_buffer) >= BUFFER_LIMIT) or \
                      (len(word_buffer) > 0 and (time.time() - last_activity_time > TIMEOUT_SECONDS))

        if should_send:
            msg = " ".join(word_buffer).strip()
            if msg == last_sent_message:
                print(f"Duplicate ignored: {msg}")
                word_buffer = []
            elif msg:
                final_message = msg
                last_sent_message = msg
                word_buffer = []
                sequence = []
                try:
                    supabase.table('conversation_logs').insert({
                        "message_content": msg, "sender_role": "signer"
                    }).execute()
                except Exception: pass

        return jsonify({
            "predicted_char": predicted_text,
            "current_word": current_word,
            "buffer_preview": " ".join(word_buffer),
            "buffer_count": f"{len(word_buffer)} / {BUFFER_LIMIT}",
            "processed_image": processed_image_b64,
            "final_message": final_message
        })

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global current_word, word_buffer, last_sent_message, sequence
    current_word = ""
    word_buffer = []
    sequence = []
    last_sent_message = ""
    return jsonify({"status": "cleared"})

if __name__ == '__main__':
    print("Server starting on Port 5000...")
    app.run(host='0.0.0.0', port=5000)