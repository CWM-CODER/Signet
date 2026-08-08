import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
import pickle
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report


DATA_PATH = r"D:/SignLanguageProject/archive/Word_Training_Data_Normalized"
MODEL_PATH = r"models/action_lstm.pth"
LABEL_PATH = r"models/action_labels.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Training on Device: {DEVICE}")
print("📂 Loading Sequence Data...")

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: '{DATA_PATH}' not found. Did you run process_wlasl.py?")
    sys.exit(1)

actions = np.array(os.listdir(DATA_PATH))
print(f"   Classes Found ({len(actions)}): {actions}")

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

print("   Reading files...")
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    file_list = os.listdir(action_path)
    
    if len(file_list) == 0: continue
        
    for file_name in file_list:
        if file_name.endswith('.npy'):
            window = np.load(os.path.join(action_path, file_name))
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences) 
y = np.array(labels)

if len(X) == 0:
    print("❌ Error: No data found.")
    sys.exit(1)

print(f"✅ Data Loaded. Shape: {X.shape}") 

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f"⚖️  Class Weights applied: {class_weights}")

X_train_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
y_train_tensor = torch.tensor(y, dtype=torch.long).to(DEVICE)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

def augment_data(inputs):
    if np.random.rand() < 0.5:
        noise = torch.randn_like(inputs) * 0.02
        inputs = inputs + noise
    if np.random.rand() < 0.5:
        scale = 0.9 + (torch.rand(1).item() * 0.2)
        inputs = inputs * scale
    return inputs

class LSTMModel(nn.Module):
    def __init__(self, input_size=150, hidden_size=64, num_layers=1, num_classes=len(actions)):
        super(LSTMModel, self).__init__()
        
        self.feature_extract = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(30), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # BLOCK 2: LSTM
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
        
        # BLOCK 3: Classifier
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

model = LSTMModel().to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights) 
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# --- 5. TRAIN ---
EPOCHS = 350
print(f"🚀 Starting Training for {EPOCHS} Epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs = augment_data(inputs)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    if (epoch + 1) % 50 == 0:
        avg_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")

# --- 6. DEBUGGING REPORT ---
print("\n📊 Generating Debug Report...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in train_loader: 
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n--- Confusion Matrix (Actual vs Predicted) ---")
print(confusion_matrix(all_labels, all_preds))
print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=actions))

# --- 7. SAVE ---
if not os.path.exists("models"): os.makedirs("models")
torch.save(model.state_dict(), MODEL_PATH)
with open(LABEL_PATH, "wb") as f:
    pickle.dump(actions, f)

print(f"\n✅ Model saved to '{MODEL_PATH}'")