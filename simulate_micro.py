import os
import torch
from torch.utils.data import DataLoader
from model import MicroAutoencoder
from dataset import RoomDataset, get_transforms
from utils import get_device
from server import Aggregator
from client import FLClient

def run_micro_simulation():
    print("--- Starting ESP32 Micro Client Simulation ---")
    data_path = "res/videos/salon_corner/sampled/normal"
    if not os.path.exists(data_path):
        print("Data not found. Run extract_samples.sh first.")
        return

    device = get_device()
    print(f"Device: {device}")
    
    # 1. Setup Micro Model (32x32 Input)
    print("Initializing MicroAutoencoder (DW/PW Convs)...")
    micro_model = MicroAutoencoder().to(device)
    
    # Count parameters
    params = sum(p.numel() for p in micro_model.parameters())
    print(f"Micro Model Parameters: {params}")
    # Estimate typical activation memory for 32x32 float32
    # Input: 3x32x32 = 3KB
    # Lay1: 8x32x32 = 8KB
    # Lay2: 8x16x16 = 2KB
    # Total active memory is very low (<50KB), fitting easily in 512KB SRAM.
    
    # 2. Setup Data (Resize to 32x32)
    # Note: get_transforms accepts size arg
    transforms = get_transforms(img_size=32)
    dataset = RoomDataset(data_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # Batch 1 for micro simulation
    
    # 3. Simulate Training
    client = FLClient(client_id="esp32", model=micro_model, train_loader=dataloader)
    
    print("\n--- Training Micro Client ---")
    initial_loss = client.train(epochs=0)[1] # Just evaluate
    print(f"Initial Loss (Random Weights): {initial_loss:.4f}")
    
    updated_weights, final_loss = client.train(epochs=5)
    print(f"Final Loss (After 5 Epochs): {final_loss:.4f}")
    
    # 4. Anomaly Test
    anomaly_path = "res/videos/salon_corner/sampled/anomaly"
    if os.path.exists(anomaly_path):
        anomaly_dataset = RoomDataset(anomaly_path, transform=transforms)
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True)
        
        scores = []
        for img in anomaly_loader:
            s, _ = client.detect_anomaly(img)
            scores.append(s)
            
        print(f"Avg Anomaly Score: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    run_micro_simulation()
