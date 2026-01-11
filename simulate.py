import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import shutil
from torch.utils.data import DataLoader, Subset
import random
import copy

from model import SimpleAutoencoder
from dataset import RoomDataset, get_transforms
from utils import get_device
from server import Aggregator
from client import FLClient

def setup_real_data(data_path, num_clients):
    """
    Loads one dataset and splits it into num_clients partitions.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Real data path not found: {data_path}")
    
    full_dataset = RoomDataset(data_path, transform=get_transforms())
    total_size = len(full_dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    split_size = total_size // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start = i * split_size
        end = (i + 1) * split_size if i < num_clients - 1 else total_size
        client_indices = indices[start:end]
        client_datasets.append(Subset(full_dataset, client_indices))
        
    return client_datasets, full_dataset

def run_simulation(num_clients=5, rounds=5):
    print(f"--- Starting FL Simulation (Clients: {num_clients}) ---")
    
    # Paths
    normal_data_path = "res/videos/salon_corner/sampled/normal"
    anomaly_data_path = "res/videos/salon_corner/sampled/anomaly"
    
    device = get_device()
    print(f"Device: {device}")

    # 1. Setup Data
    print(f"Loading real data from {normal_data_path}...")
    try:
        client_datasets, _ = setup_real_data(normal_data_path, num_clients)
    except FileNotFoundError:
        print("Real data not found! Please run extract_samples.sh first.")
        return

    # 2. Setup Components
    global_model = SimpleAutoencoder().to(device)
    aggregator = Aggregator(global_model)
    
    clients = []
    for i in range(num_clients):
        # Batch size can be small for RPi simulation
        dataloader = DataLoader(client_datasets[i], batch_size=4, shuffle=True)
        client = FLClient(client_id=i, model=global_model, train_loader=dataloader)
        clients.append(client)
        print(f"Client {i} initialized with {len(client_datasets[i])} samples.")
        
    # 3. FL Loop (FedALA)
    print("\n--- Federated Learning Phase (FedALA) ---")
    for r in range(rounds):
        client_updates = []
        round_losses = []
        
        current_global_model = aggregator.get_global_model()
        
        for client in clients:
            # Personalize/Adapt
            client.adapt_global_model(current_global_model, ala_epochs=1)
            
            # Local Training
            updated_weights, loss = client.train(epochs=2)
            client_updates.append(updated_weights)
            round_losses.append(loss)
            
        aggregator.aggregate(client_updates)
        avg_loss = sum(round_losses) / len(round_losses)
        print(f"Round {r+1}/{rounds} - Avg Loss: {avg_loss:.4f}")

    # 4. Anomaly Detection Test
    print("\n--- Anomaly Detection Test ---")
    if os.path.exists(anomaly_data_path):
        anomaly_dataset = RoomDataset(anomaly_data_path, transform=get_transforms())
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=1, shuffle=True)
        
        # Test just the first client as an example (they are all similar in this simulation)
        tester_client = clients[0]
        print(f"Testing Client 0 on {len(anomaly_dataset)} anomaly frames...")
        
        anomaly_scores = []
        for img in anomaly_loader:
             score, is_anomaly = tester_client.detect_anomaly(img, threshold=0.03) # Lower threshold for real data maybe?
             anomaly_scores.append(score)
        
        avg_anomaly_score = sum(anomaly_scores) / len(anomaly_scores)
        print(f"Average Anomaly Score: {avg_anomaly_score:.4f}")
        
        # Compare with Normal Score
        normal_scores = []
        # Use a few training samples
        for i in range(min(10, len(client_datasets[0]))):
            img = client_datasets[0][i].unsqueeze(0).to(device)
            score, _ = tester_client.detect_anomaly(img)
            normal_scores.append(score)
        avg_normal_score = sum(normal_scores) / len(normal_scores)
        print(f"Average Normal Score: {avg_normal_score:.4f}")
        
        if avg_anomaly_score > avg_normal_score:
            print("SUCCESS: Anomaly score is higher than normal score!")
        else:
            print("WARNING: Anomaly score is not distinct enough. Model might need more training or threshold adjustment.")
            
    else:
        print("Anomaly data path not found. Skipping test.")

if __name__ == "__main__":
    run_simulation()
