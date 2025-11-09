import pandas as pd
from PIL import Image
import numpy as np
import os
import random
from pathlib import Path

# File locations
CSV_PATH = "car_crash_data.csv"  # CSV file in the same directory
IMG_FOLDER = "c:\\Users\\prana\\OneDrive\\Desktop\\Car Crash Model\\dataset"  # Absolute path to dataset
BATCH_SIZE = 5
IMG_SIZE = (128, 128)

def generate_csv_from_directory():
    """Generate CSV file from images in the dataset directory"""
    images = []
    # Get all jpg files from the directory
    for img_file in Path(IMG_FOLDER).glob("*.jpg"):
        images.append({
            'image': img_file.name,
            'weather': random.choice(['clear', 'rainy', 'cloudy', 'snowy']),  # Simulated data
            'time_of_day': random.choice(['day', 'night', 'dawn', 'dusk']),
            'road_type': random.choice(['urban', 'highway', 'rural']),
            'speed': random.randint(20, 120),
            'severity': random.choice(['minor', 'moderate', 'severe'])
        })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(images)
    df.to_csv(CSV_PATH, index=False)
    return df

# Generate or read CSV file
if not os.path.exists(CSV_PATH):
    df = generate_csv_from_directory()
else:
    df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()
image_files = df['image'].dropna().tolist()

def get_random_image():
    """Get a random image and its metadata from the dataset"""
    if not image_files:
        return None, None
    
    # Pick a random image
    img_name = random.choice(image_files)
    img_data = df[df['image'] == img_name].iloc[0].to_dict()
    
    img_path = os.path.join(IMG_FOLDER, img_name)
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        return img, img_data
    return None, None

def get_image_batch(start, batch_size=BATCH_SIZE):
    """Get a batch of images for training"""
    batch_imgs = []
    batch_files = image_files[start:start+batch_size]
    for fname in batch_files:
        img_path = os.path.join(IMG_FOLDER, fname)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            img_arr = np.array(img) / 255.0
            batch_imgs.append(img_arr)
    return np.stack(batch_imgs) if batch_imgs else None, batch_files

def get_training_progress(epoch, total_epochs, batch_idx, total_batches):
    """Generate training progress information"""
    return {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'batch': batch_idx,
        'total_batches': total_batches,
        'progress': (epoch * total_batches + batch_idx) / (total_epochs * total_batches) * 100
    }

if __name__ == "__main__":
    # Test random image selection
    img, metadata = get_random_image()
    if img and metadata:
        print("Selected image:", metadata['image'])
        print("Metadata:", metadata)
        img.show()  # Display the image (for testing)
        
    # Test batch loading
    imgs, files = get_image_batch(0, BATCH_SIZE)
    print("\nProcessed files:", files)
    print("Image batch shape:", imgs.shape if imgs is not None else "No images found")
