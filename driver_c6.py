import argparse
import time
import struct
import os
import cv2
import numpy as np
import serial

# Protocol Constants
MAGIC = b'NFL0'
PAYLOAD_Floats = 100  # 10x10 grid
PAYLOAD_BYTES = PAYLOAD_Floats * 4

# Cache for frame differencing
_prev_frame_cache = {}

def preprocess_image(image_path, img_w=360, img_h=240, grid_x=10, grid_y=10, use_motion=False):
    """
    Extract features for anomaly detection.
    
    If use_motion=True (default): Uses frame differencing to capture motion patterns.
    - Computes absolute difference between current and previous frame
    - Extracts 5x5 grid features from motion map
    - Much better for detecting unusual motion (bikes, carts in UCSD)
    
    If use_motion=False: Uses static average intensity per grid cell.
    """
    if not os.path.exists(image_path):
        return None
        
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    img = cv2.resize(img, (img_w, img_h))
    
    # Get clip/sequence identifier from path (for frame differencing cache)
    clip_id = os.path.dirname(image_path)
    
    if use_motion:
        # Motion-based features (frame differencing)
        if clip_id in _prev_frame_cache:
            prev = _prev_frame_cache[clip_id]
            # Absolute difference captures motion
            diff = cv2.absdiff(img, prev).astype(np.float32)
        else:
            # First frame: use zeros (no motion detected yet)
            diff = np.zeros_like(img, dtype=np.float32)
        
        # Update cache with current frame
        _prev_frame_cache[clip_id] = img.copy()
        
        # Apply slight blur to reduce noise
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Extract grid features from motion map
        features = []
        block_w = img_w // grid_x
        block_h = img_h // grid_y
        
        for gy in range(grid_y):
            for gx in range(grid_x):
                start_y = gy * block_h
                start_x = gx * block_w
                block = diff[start_y:start_y+block_h, start_x:start_x+block_w]
                # Normalize motion intensity to 0-1
                avg_motion = np.mean(block) / 255.0
                features.append(avg_motion)
    else:
        # Static features (original implementation)
        features = []
        block_w = img_w // grid_x
        block_h = img_h // grid_y
        
        for gy in range(grid_y):
            for gx in range(grid_x):
                start_y = gy * block_h
                start_x = gx * block_w
                block = img[start_y:start_y+block_h, start_x:start_x+block_w]
                avg = np.mean(block)
                features.append(avg / 255.0)
            
    return features

def reset_motion_cache():
    """Clear the frame differencing cache (call between clips/sequences)."""
    global _prev_frame_cache
    _prev_frame_cache = {}

def send_frame(ser, features):
    if len(features) != PAYLOAD_Floats:
        print(f"Error: Expected {PAYLOAD_Floats} features, got {len(features)}")
        return

    # Packet: MAGIC + LEN (u16_le) + DATA (25 * float32_le)
    header = MAGIC + struct.pack('<H', PAYLOAD_BYTES)
    payload = struct.pack(f'<{PAYLOAD_Floats}f', *features)
    
    ser.write(header + payload)
    
    # Wait for OK/ERR response with timeout
    time.sleep(0.05) 
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"C6: {line}")

def main():
    parser = argparse.ArgumentParser(description="Feed images to ESP32-C6 Anomaly Detector")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--dir", default="data/ucsdpeds/vidf", help="Directory containing frames")
    parser.add_argument("--interval", type=float, default=0.1, help="Interval between frames")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = parser.parse_args()

    print(f"Opening Serial {args.port} @ {args.baud}...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Could not open serial port: {e}")
        print("Note: If you don't have a C6 connected, this script just prints what it WOULD send.")
        ser = None
        
    # Get images recursively
    images = []
    for root, dirs, files in os.walk(args.dir):
        for f in sorted(files):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                images.append(os.path.join(root, f))
    
    # Sort again to ensure order across directories if needed, though usually sequence matters per dir
    images.sort()
    
    if args.limit > 0 and len(images) > args.limit:
        print(f"Limiting to first {args.limit} images.")
        images = images[:args.limit]

    if not images:
        print(f"No images found in {args.dir}")
        
        # Fallback to dummy generation for testing loop logic
        print("Using dummy features for testing...")
        features = [0.5] * PAYLOAD_Floats
        if ser:
            send_frame(ser, features)
        return

    print(f"Found {len(images)} images.")
    
    for img_path in images:
        features = preprocess_image(img_path)
        if features:
            print(f"Sending {os.path.basename(img_path)}...")
            if ser:
                send_frame(ser, features)
            else:
                print(f"[Simulation] Payload: {features[:5]}... (First 5)")
                
        time.sleep(args.interval)

if __name__ == "__main__":
    main()
