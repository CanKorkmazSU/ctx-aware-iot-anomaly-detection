# Raspberry Pi Federated Learning Anomaly Detection

This project implements a privacy-preserving anomaly detection system using **Federated Learning (FL)**. It is designed for resource-constrained edge devices like Raspberry Pi, using a lightweight Convolutional Autoencoder and **FedALA (Adaptive Local Aggregation)** for personalization.

## Features
- **Federated Learning**: Privacy-preserving training across multiple "room" clients.
- **FedALA (New)**: Channel-wise Adaptive Local Aggregation. Clients learn to blend global and local weights (`W_new = alpha * W_global + (1-alpha) * W_local`) for better personalization.
- **Micro Client (New)**: Specialized `MicroAutoencoder` for ESP32 (512KB SRAM) using Depthwise Separable Convolutions (~800 params).
- **No Heavy Dependencies**: Custom `dataset.py` handles transforms without `torchvision`.

## Project Structure

- `model.py`: **SimpleAutoencoder** (RPi) and **MicroAutoencoder** (ESP32).
- `dataset.py`: Custom image loader and transforms.
- `client.py`: FL Client with **FedALA** implementation using `torch.func`.
- `server.py`: FL Server aggregator.
- `camera.py`/`detect.py`: Real-time deployment scripts.
- `simulate.py`: Simulates FL with 5 clients using real data.
- `simulate_micro.py`: Simulates ESP32 Micro Client performance.
- `extract_samples.sh`: Helper to prepare dataset from videos.

## Setup

1.  **Create and Source Virtual Environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation
To run simulations with real data, extract frames from your videos:
```bash
chmod +x extract_samples.sh
# Syntax: ./extract_samples.sh <video> <fps> <output_dir>
./extract_samples.sh res/videos/salon_corner/salon_corner_normal.mp4 10 res/videos/salon_corner/sampled/normal
./extract_samples.sh res/videos/salon_corner/salon_corner_anormal.mp4 2 res/videos/salon_corner/sampled/anomaly
```

### 2. Run Simulation (Raspberry Pi Model)
Simulates 5 clients using FedALA on the extracted data:
```bash
python3 simulate.py
```
*   **Result**: Compares Anomaly Score vs Normal Score on real data.

### 3. Run Simulation (ESP32 Micro Model)
Simulates a constrained client with the `MicroAutoencoder` (32x32 input):
```bash
python3 simulate_micro.py
```
*   **Result**: Verifies model fits in memory and can detect anomalies.

### 4. Real-time Detection (On Device)
To run live implementation on a Raspberry Pi:
```bash
python3 detect.py --threshold 0.02
```

## Requirements
- Python 3.7+
- PyTorch (CPU version is fine for RPi)
- NumPy, Pillow, OpenCV
