import torch
import time
import argparse
from model import SimpleAutoencoder
from dataset import get_transforms
from camera import Camera
from utils import get_device

def main(model_path, threshold):
    device = get_device()
    print(f"Using device: {device}")
    
    # Load model
    model = SimpleAutoencoder()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    model.to(device)
    model.eval()
    
    transforms = get_transforms()
    camera = Camera(0)
    
    try:
        while True:
            try:
                img = camera.capture_frame()
                img_tensor = transforms(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    loss = torch.nn.functional.mse_loss(output, img_tensor)
                    score = loss.item()
                    
                is_anomaly = score > threshold
                status = "ANOMALY!" if is_anomaly else "Normal"
                print(f"Score: {score:.6f} | Status: {status}")
                
                time.sleep(0.5) # Throttle
                
            except Exception as e:
                print(f"Error: {e}")
                break
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        camera.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Path to saved model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.02, help="Anomaly threshold")
    args = parser.parse_args()
    
    main(args.model, args.threshold)
