"""Hybrid FL Simulation - Real UCSD Data.

Simple, human-readable script to simulate Federated Learning on the UCSD Anomaly Dataset.
Uses real ground-truth labels (pixel masks) to evaluate performance.
"""

from __future__ import annotations

import argparse
import os
from PIL import Image
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

# --- Configuration ---
PAYLOAD_FLOATS = 100  # 10x10 grid features

# --- Model (Scaled for ESP32 512KB SRAM) ---
class NanoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 48)  # Input: 10x10 grid
        self.fc2 = nn.Linear(48, 100)  # Output: Reconstruction
    
    def forward(self, x):
        # Simple Autoencoder: Squash input to 48 dims, then reconstruct
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# --- Feature Extraction (Host Side) ---
def preprocess_image(image_path, grid_x=10, grid_y=10):
    """Extract simple 10x10 grid features from an image."""
    if not os.path.exists(image_path): return None
    
    # Read as Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # Resize to standard frame info
    img = cv2.resize(img, (160, 120))
    img = img.astype(np.float32) / 255.0
    
    # Grid Average Pooling (Simple & Fast)
    # Calculate average intensity in each 10x10 cell
    h, w = img.shape
    bh, bw = h // grid_y, w // grid_x
    
    flat_features = []
    for y in range(grid_y):
        for x in range(grid_x):
            block = img[y*bh:(y+1)*bh, x*bw:(x+1)*bw]
            avg = np.mean(block)
            flat_features.append(float(avg))
            
    return flat_features

# --- Data Loading (Real UCSD Data) ---
def load_ucsd_real(base_dir, max_train=2000, max_test=500):
    """Load real training frames and test frames with ground truth labels."""
    train_features = []
    test_features = []
    test_labels = [] # 0 = Normal, 1 = Anomaly
    
    print(f"Loading data from {base_dir}...")

    # 1. Load Training Data (All Normal)
    train_dir = os.path.join(base_dir, "Train")
    if os.path.isdir(train_dir):
        # Iterate over clip folders (Train001, Train002...)
        for clip in sorted(os.listdir(train_dir)):
            clip_path = os.path.join(train_dir, clip)
            if not os.path.isdir(clip_path): continue
            
            for f in sorted(os.listdir(clip_path)):
                if not f.endswith('.tif'): continue
                if len(train_features) >= max_train: break
                
                path = os.path.join(clip_path, f)
                feats = preprocess_image(path)
                if feats: train_features.append(feats)
                
    # 2. Load Test Data (Normal + Anomalies w/ Masks)
    test_dir = os.path.join(base_dir, "Test")
    if os.path.isdir(test_dir):
        # Iterate over test clips (Test001, Test002...)
        # Skip _gt folders
        clips = [c for c in sorted(os.listdir(test_dir)) 
                 if os.path.isdir(os.path.join(test_dir, c)) and not c.endswith('_gt')]
                 
        for clip in clips:
            clip_path = os.path.join(test_dir, clip)
            gt_path = os.path.join(test_dir, f"{clip}_gt")
            
            for f in sorted(os.listdir(clip_path)):
                if not f.endswith('.tif'): continue
                if len(test_features) >= max_test: break
                
                path = os.path.join(clip_path, f)
                feats = preprocess_image(path)
                if not feats: continue
                
                # Check Ground Truth Label
                is_anomaly = 0
                mask_name = f.replace('.tif', '.bmp')
                mask_path = os.path.join(gt_path, mask_name)
                
                if os.path.exists(mask_path):
                    # If mask has any white pixels, it's an anomaly
                    mask = np.array(Image.open(mask_path))
                    if np.any(mask > 0):
                        is_anomaly = 1
                
                test_features.append(feats)
                test_labels.append(is_anomaly)

    return train_features, test_features, test_labels

# --- Simulation Client ---
class SimClient:
    def __init__(self, name, data, lr):
        self.name = name
        self.data = np.array(data, dtype=np.float32)
        self.model = NanoMLP()
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
    def set_weights(self, global_weights):
        # Load weights from list into model
        if not global_weights: return
        with torch.no_grad():
            ptr = 0
            for p in self.model.parameters():
                num = p.numel()
                p.copy_(torch.tensor(global_weights[ptr:ptr+num]).reshape(p.shape))
                ptr += num
                
    def get_weights(self):
        # Return weights as flat list
        return np.concatenate([p.detach().numpy().flatten() for p in self.model.parameters()]).tolist()
        
    def train(self, epochs, batch_size=32):
        if len(self.data) == 0: return 0.0
        
        self.model.train()
        x = torch.tensor(self.data)
        total_loss = 0
        batches = 0
        
        for _ in range(epochs):
            # Simple batch shuffle
            indices = torch.randperm(len(x))
            for i in range(0, len(x), batch_size):
                batch = x[indices[i:i+batch_size]]
                
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.loss_fn(output, batch) # Autoencoder: Input is Target
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batches += 1
                
        return total_loss / max(1, batches)

# --- Metrics Calculation ---
def evaluate(model, test_X, test_Y):
    if len(test_X) == 0: return {}
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.array(test_X, dtype=np.float32))
        recon = model(x)
        # Anomaly Score = Reconstruction Error (MSE)
        scores = torch.mean((x - recon)**2, dim=1).numpy()
        
    auc = roc_auc_score(test_Y, scores) if len(set(test_Y)) > 1 else 0.5
    
    # Optimal Threshold (Youden's J)
    fpr, tpr, thresholds = roc_curve(test_Y, scores)
    optimal_idx = np.argmax(tpr - fpr)
    threshold = thresholds[optimal_idx]
    
    preds = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(test_Y, preds, average='binary', zero_division=0)
    
    # Debug: Average scores
    norm_scores = scores[np.array(test_Y) == 0]
    anom_scores = scores[np.array(test_Y) == 1]
    
    return {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'thresh': threshold,
        'norm_avg': np.mean(norm_scores) if len(norm_scores) else 0,
        'anom_avg': np.mean(anom_scores) if len(anom_scores) else 0
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--dataset", default="UCSDped2")
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()
    
    # 1. Load Data
    base_path = f"data/UCSD_Anomaly_Dataset.v1p2/{args.dataset}"
    train, test, labels = load_ucsd_real(base_path)
    print(f"Data Loaded: {len(train)} Train samples, {len(test)} Test samples ({sum(labels)} Anomalies)")
    
    if len(train) == 0:
        print("Error: No training data found.")
        return

    # 2. Setup Simulation
    # Split training data among 10 clients
    chunk_size = len(train) // 10
    clients = [SimClient(f"C{i}", train[i*chunk_size:(i+1)*chunk_size], args.lr) for i in range(10)]
    
    global_client = SimClient("Global", [], args.lr)
    global_weights = global_client.get_weights()
    
    print(f"\n{'Round':<5} | {'Loss':<8} | {'AUC':<6} | {'F1':<6} | {'NormErr':<8} | {'AnomErr':<8}")
    print("-" * 55)
    
    # 3. Training Loop
    for r in range(args.rounds):
        round_losses = []
        
        # A. Local Training
        updated_weights = []
        for client in clients:
            client.set_weights(global_weights)
            loss = client.train(epochs=5)
            round_losses.append(loss)
            updated_weights.append(client.get_weights())
            
        # B. Aggregation (FedAvg)
        # Average all weights
        avg_weights = np.mean(updated_weights, axis=0).tolist()
        global_client.set_weights(avg_weights)
        global_weights = avg_weights
        
        # C. Evaluation
        metrics = evaluate(global_client.model, test, labels)
        
        print(f"{r+1:<5} | {np.mean(round_losses):<8.4f} | {metrics['auc']:<6.4f} | {metrics['f1']:<6.4f} | {metrics['norm_avg']:<8.4f} | {metrics['anom_avg']:<8.4f}")
        
    print("\nSimulation Complete.")

if __name__ == "__main__":
    main()
