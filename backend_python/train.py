import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import sys
import os

if not torch.cuda.is_available():
    print("❌ CRITICAL ERROR: No NVIDIA GPU detected!")
    print("This script is configured to ONLY run on GPU.")
    print("Please check your CUDA installation.")
    sys.exit(1)

device = torch.device("cuda")
print(f"✅ GPU Detected & Locked: {torch.cuda.get_device_name(0)}")
print(f"   Memory Usage: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

CSV_FILE = "landmark_data.csv"
BATCH_SIZE = 64
EPOCHS = 100  
LEARNING_RATE = 0.001

class LandmarkDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

print("📂 Loading skeleton data from CSV...")
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: '{CSV_FILE}' not found.")
    print("Please run 'convert_dataset.py' first to create this file!")
    sys.exit(1)

try:
    df = pd.read_csv(CSV_FILE)
except Exception as e:
    print(f"❌ Error reading CSV: {e}")
    sys.exit(1)

X = df.iloc[:, :-1].values 
y = df.iloc[:, -1].values 

# Encode Labels (A -> 0, B -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

with open("labels.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"✅ Labels encoded and saved to 'labels.pkl'. Found {num_classes} classes.")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

train_data = LandmarkDataset(X_train, y_train)
test_data = LandmarkDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

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

model = LandmarkNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"🚀 Starting GPU Training for {EPOCHS} Epochs...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

torch.save(model.state_dict(), "model.pth")
print("✅ Training Complete. Model saved as 'model.pth'")