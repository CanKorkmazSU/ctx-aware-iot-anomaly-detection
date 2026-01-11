import os
import argparse
import time
import struct
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import serial
from driver_c6 import preprocess_image

# --- Constants & Configuration ---
MAGIC_NFL0 = b'NFL0'  # Data
MAGIC_NFLW = b'NFLW'  # Set Weights: Host -> C6
MAGIC_NFLR = b'NFLR'  # Get Weights: Host -> C6
PAYLOAD_FLOATS = 25
PAYLOAD_BYTES = PAYLOAD_FLOATS * 4

# --- 1. Model Definition (PyTorch matching ESP32 Genann) ---
class NanoMLP(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=12, output_dim=25):
        super(NanoMLP, self).__init__()
        # Genann uses sigmoid activation on hidden
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        # Genann output is usually linear unless configured otherwise, 
        # but in our C++ model.h we compute MSE manually.
        # Let's assume linear output for reconstruction, or sigmoid if normalized.
        # Genann default activation is sigmoid.
        x = torch.sigmoid(self.fc2(x)) 
        return x

# --- 2. Protocols ---
def serial_set_weights(ser, weights_list):
    """
    Send weights to C6. 
    Protocol: NFLW + LEN(u16) + FLAGS(float32...flat)
    """
    if not ser: return
    
    flat_weights = np.array(weights_list, dtype=np.float32)
    payload_len = len(flat_weights) * 4
    
    header = MAGIC_NFLW + struct.pack('<H', payload_len)
    payload = flat_weights.tobytes()
    
    SER_MAX_CHUNK = 256
    
    # Send Header
    ser.write(header)
    
    # Chunked Write for Payload
    total_sent = 0
    while total_sent < len(payload):
        chunk = payload[total_sent : total_sent + SER_MAX_CHUNK]
        ser.write(chunk)
        total_sent += len(chunk)
        time.sleep(0.005) # Tiny sleep to let C6 process
    
    # Wait for ack
    start = time.time()
    while time.time() - start < 5.0:
        if ser.in_waiting:
            raw_line = ser.readline()
            try:
                line = raw_line.decode('utf-8', errors='ignore').strip()
                print(f"[DEBUG] RX: {line}") # Debug print
                if "OK SET_WEIGHTS" in line:
                    return True
            except:
                print(f"[DEBUG] RX (bin): {raw_line}")
                
    print("Warning: C6 did not ack SET_WEIGHTS")
    return False

def serial_get_weights(ser):
    """
    Request weights from C6.
    Protocol: NFLR -> C6 responds with NFLW + LEN + DATA
    """
    if not ser: return None
    
    # Clear any pending junk in buffers
    ser.reset_input_buffer()
    time.sleep(0.05)
    
    ser.write(MAGIC_NFLR)
    
    # Seek for response header (NFLW + LEN)
    # But we need to find NFLW in the stream first
    start = time.time()
    
    # 1. Find magic NFLW (could be preceded by garbage)
    search_buf = b''
    magic_idx = -1
    while magic_idx < 0:
        if time.time() - start > 5.0:
            print("Timeout waiting for weights magic")
            return None
        if ser.in_waiting:
            search_buf += ser.read(ser.in_waiting)
            magic_idx = search_buf.find(MAGIC_NFLW)
        else:
            time.sleep(0.01)
            
    # 2. Read rest of header (LEN = 2 bytes) if not already buffered
    # After magic, we need 2 more bytes for length
    header_rest_start = magic_idx + 4
    while len(search_buf) < header_rest_start + 2:
        if time.time() - start > 5.0:
            print("Timeout waiting for weights length")
            return None
        if ser.in_waiting:
            search_buf += ser.read(ser.in_waiting)
        else:
            time.sleep(0.01)
            
    len_bytes = search_buf[header_rest_start : header_rest_start + 2]
    payload_len = struct.unpack('<H', len_bytes)[0]
    
    # Sanity check: Expected payload is 637 floats * 4 bytes = 2548 bytes
    EXPECTED_PAYLOAD = 637 * 4
    if payload_len != EXPECTED_PAYLOAD:
        print(f"[C6] Invalid payload length: {payload_len} (expected {EXPECTED_PAYLOAD})")
        return None
        
    expected_floats = payload_len // 4
    
    # 3. Check how much data we already have after the header
    header_end = header_rest_start + 2
    already_have = search_buf[header_end:]
    data = already_have
    
    # Read Data (Robust Loop)
    ser.timeout = 2.0
    last_recv_time = time.time()
    
    while len(data) < payload_len:
        if time.time() - start > 15.0:
            print(f"Timeout/Error reading weight payload: Got {len(data)}/{payload_len}")
            return None
        remaining = payload_len - len(data)
        chunk = ser.read(remaining)
        if chunk:
            data += chunk
            last_recv_time = time.time()
        else:
            if time.time() - last_recv_time > 3.0:
                 print(f"Stall reading weight payload: Got {len(data)}/{payload_len}")
                 return None
             
    floats = struct.unpack(f'<{expected_floats}f', data)
    return list(floats)

def serial_send_data(ser, features):
    """Train on one sample."""
    if not ser: return
    # NFL0 + LEN + DATA
    payload_len = 25 * 4
    header = MAGIC_NFL0 + struct.pack('<H', payload_len)
    payload = struct.pack(f'<25f', *features)
    ser.write(header + payload)
    time.sleep(0.02) # Give C6 time to process

# --- 3. Clients ---
class SimClient:
    def __init__(self, id, data):
        self.id = id
        self.data = data # List of feature vectors (lists of 25 floats)
        self.model = NanoMLP()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()
        
    def set_weights(self, global_weights):
        # Load flat weights into state dict
        state_dict = self.model.state_dict()
        ptr = 0
        
        # We need to map flat list back to tensors
        # Genann layout order: Hidden Unit 0 (Input weights...), Hidden Unit 1... 
        # PyTorch defaults: weight [out_features, in_features], bias [out_features]
        # Genann: Weights are stored per-neuron. 
        #   ann->weight + (neuron_idx * (inputs + 1))
        #   Layout: [w0, w1... wIn, bias] for Neuron 0
        
        # Mapping PyTorch Linear to Genann:
        
        # Layer 1
        w1 = state_dict['fc1.weight'] # [12, 25]
        b1 = state_dict['fc1.bias']   # [12]
        
        # Layer 2
        w2 = state_dict['fc2.weight'] # [25, 12]
        b2 = state_dict['fc2.bias']   # [25]
        
        # This mapping is complex. For simplicity in simulation, we will TRUST that 
        # if SimClient and C6 Client communicate using the SAME flat format, averaging works.
        # So SimClient effectively needs to just hold a flat buffer or use PyTorch carefully.
        # Let's stick to PyTorch but ensure export/import matches Genann layout.
        
        # Import:
        # Layer 1 (Hidden)
        for i in range(12): # Hidden Dim
            # Genann: [25 inputs..., bias]
            chunk_size = 25 + 1
            chunk = global_weights[ptr : ptr + chunk_size]
            ptr += chunk_size
            
            with torch.no_grad():
                self.model.fc1.weight[i, :] = torch.tensor(chunk[:25], dtype=torch.float32)
                self.model.fc1.bias[i] = torch.tensor(chunk[25], dtype=torch.float32)
                
        # Layer 2 (Output)
        for i in range(25): # Output Dim
            # Genann: [12 inputs..., bias]
            chunk_size = 12 + 1
            chunk = global_weights[ptr : ptr + chunk_size]
            ptr += chunk_size
            
            with torch.no_grad():
                self.model.fc2.weight[i, :] = torch.tensor(chunk[:12], dtype=torch.float32)
                self.model.fc2.bias[i] = torch.tensor(chunk[12], dtype=torch.float32)

    def get_weights(self):
        # Export to Genann Flat Format
        flat = []
        
        # Layer 1
        w1 = self.model.fc1.weight.detach().numpy()
        b1 = self.model.fc1.bias.detach().numpy()
        for i in range(12):
            flat.extend(w1[i, :])
            flat.append(b1[i])
            
        # Layer 2
        w2 = self.model.fc2.weight.detach().numpy()
        b2 = self.model.fc2.bias.detach().numpy()
        for i in range(25):
            flat.extend(w2[i, :])
            flat.append(b2[i])
            
        return flat

    def train(self, epochs=1):
        self.model.train()
        losses = []
        # Convert data to tensors
        tensor_data = torch.tensor(self.data, dtype=torch.float32)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            # Reconstruction Autoencoder: Input -> Output
            outputs = self.model(tensor_data)
            loss = self.loss_fn(outputs, tensor_data)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

class HardwareClient:
    def __init__(self, port, data):
        self.port = port
        self.data = data
        self.ser = None
        self.ensure_connection()
            
    def ensure_connection(self):
        if self.ser and self.ser.is_open:
            return
        
        print(f"[C6] Connecting to {self.port}...")
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=1)
            # Flush junk
            self.ser.read_all()
            print("[C6] Connected.")
        except Exception as e:
            print(f"[C6] Connection failed: {e}")
            self.ser = None

    def set_weights(self, global_weights):
        self.ensure_connection()
        if self.ser:
            try:
                success = serial_set_weights(self.ser, global_weights)
                if not success:
                    # Logic error, not necessarily disconnect
                    pass
            except serial.SerialException as e:
                print(f"[C6] Error during set_weights: {e}")
                self.ser.close()
                self.ser = None

    def get_weights(self):
        self.ensure_connection()
        if self.ser:
            try:
                w = serial_get_weights(self.ser)
                if w: return w
            except serial.SerialException as e:
                print(f"[C6] Error during get_weights: {e}")
                self.ser.close()
                self.ser = None
                
        # If no device or failed, return last known dummy (Critical for simulation continuity)
        # Ideally return None to signal 'drop out' this round, but FedAvg might crash if list length changes?
        # Simulation loop handles None return by skipping.
        return None

    def train(self, epochs=1):
        self.ensure_connection()
        if not self.ser: return 0.0
        
        try:
            for _ in range(epochs):
                for feat in self.data:
                    serial_send_data(self.ser, feat)
        except serial.SerialException as e:
             print(f"[C6] Error during train (send data): {e}")
             self.ser.close()
             self.ser = None
             
        return 0.0 # Unknown loss

# --- 4. Main Simulation ---
def run_hybrid_simulation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--dir", default="data/ucsdpeds/vidf")
    args = parser.parse_args()
    
    print("--- Hybrid Federated Learning (9 Sim + 1 Hardware) ---")
    
    # 1. Load Data
    print("Loading and preprocessing data...")
    all_files = []
    for root, _, files in os.walk(args.dir):
        for f in sorted(files):
            if f.endswith('.png'): all_files.append(os.path.join(root, f))
            
    # Limit for speed if too many
    # Limit for speed if too many
    all_files = all_files[:1000]
    
    # Split Train/Val (90/10)
    import random
    random.shuffle(all_files)
    val_count = int(len(all_files) * 0.1)
    train_files = all_files[val_count:]
    val_files = all_files[:val_count]
    
    print(f"Data Split: {len(train_files)} Train, {len(val_files)} Val")

    # Preprocess Train
    all_features = []
    for f in train_files:
        feats = preprocess_image(f)
        if feats: all_features.append(feats)
        
    # Preprocess Val
    val_features = []
    for f in val_files:
        feats = preprocess_image(f)
        if feats: val_features.append(feats)

    # Split Train Data among Clients
    total_clients = 10
    subset_size = len(all_features) // total_clients
    
    clients = []
    for i in range(9):
        data_slice = all_features[i*subset_size : (i+1)*subset_size]
        clients.append(SimClient(id=f"Sim_{i}", data=data_slice))
        
    hw_data = all_features[9*subset_size :]
    hw_client = HardwareClient(args.port, hw_data)
    clients.append(hw_client)
    
    print("Clients initialized.")
    
    # Helper for validation
    def validate_global(weights, val_data):
        # Temp model
        model = NanoMLP()
        # Load weights (Reuse SimClient logic or duplicate)
        # We'll instantiate a dummy client just to set weights easily
        dummy = SimClient("val", [])
        dummy.set_weights(weights)
        model = dummy.model
        model.eval()
        
        # 1. Normal Loss
        val_tensor = torch.tensor(val_data, dtype=torch.float32)
        with torch.no_grad():
            output = model(val_tensor)
            loss_normal = nn.MSELoss()(output, val_tensor).item()
            
        # 2. Anomaly Loss (Random Noise)
        # Create fake noise data
        noise = torch.rand((len(val_data), 25), dtype=torch.float32)
        with torch.no_grad():
            output_n = model(noise)
            loss_anomaly = nn.MSELoss()(output_n, noise).item()
            
        return loss_normal, loss_anomaly

    # Initialize Global Weights
    temp = SimClient("temp", [])
    global_weights = temp.get_weights()

    print(f"\n{'='*60}")
    print(f"{'Round':<6} | {'Train Loss':<12} | {'Val Loss (Ok)':<15} | {'Val Loss (Bad)':<15}")
    print(f"{'='*60}")

    # --- FL Loop ---
    for r in range(args.rounds):
        round_losses = []
        
        # Train Clients
        for client in clients:
            is_hw = isinstance(client, HardwareClient)
            # print(f"Training {client.id if not is_hw else 'C6'}...", end='', flush=True)
            
            client.set_weights(global_weights)
            loss = client.train(epochs=1)
            
            # For HW client, we don't know loss yet, assume 0 or last known
            if not is_hw: 
                round_losses.append(loss)
            
            # Aggregate partial update? No, standard FedAvg waits for all.
            # But we get weights now.
        
        # In standard FedAvg, we first collect all weights THEN average.
        # My previous loop did get_weights right after train, which is fine.
        
        # Re-collect weights properly
        collected_weights = []
        for client in clients:
             w = client.get_weights()
             if w: collected_weights.append(w)

        # Aggregate
        if collected_weights:
            arr = np.array(collected_weights)
            avg_weights = np.mean(arr, axis=0) # [637]
            global_weights = avg_weights.tolist()
        
        # Validation
        val_loss, anom_loss = validate_global(global_weights, val_features)
        avg_train = sum(round_losses) / len(round_losses) if round_losses else 0
        
        print(f"{r+1:<6} | {avg_train:<12.6f} | {val_loss:<15.6f} | {anom_loss:<15.6f}")

    print(f"{'='*60}")
    print("\nHybrid Simulation Complete.")

if __name__ == "__main__":
    run_hybrid_simulation()
