import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib

# Constants (defaults tuned for quick presentation runs)
IMG_SIZE = 224  # ResNet expects 224x224 images
# Use small batch and single epoch by default for presentations
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class
class CarCrashDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # First column contains image names
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Convert 'y'/'n' to 1/0
        label = 1 if self.data.iloc[idx, 1] == 'y' else 0
        return image, torch.tensor(label, dtype=torch.float32)

# Model definition
def get_model():
    # Use ResNet18 as base model
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

def train_model(progress_callback=None, num_epochs=DEFAULT_NUM_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    print("Starting model training...")
    print(f"Using device: {DEVICE}")
    
    # Data augmentation and preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Dataset paths
    csv_path = "c:\\Users\\prana\\OneDrive\\Desktop\\Car Crash Model\\dataset_database.csv"
    img_dir = "c:\\Users\\prana\\OneDrive\\Desktop\\Car Crash Model\\dataset"
    
    print("Loading and splitting data...")
    # Load and split data
    df = pd.read_csv(csv_path)
    # Quick sanity: drop rows with missing values
    df = df.dropna()
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['collision'])
    
    # Create temporary CSVs for train and validation sets
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    # Create datasets
    train_dataset = CarCrashDataset('train_temp.csv', img_dir, train_transform)
    val_dataset = CarCrashDataset('val_temp.csv', img_dir, val_transform)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders (use provided batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = get_model()
    model = model.to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress via callback if provided
            if progress_callback is not None:
                prog = {
                    'is_training': True,
                    'epoch': epoch + 1,
                    'total_epochs': num_epochs,
                    'batch': batch_idx + 1,
                    'total_batches': len(train_loader),
                    'progress': (epoch * len(train_loader) + (batch_idx + 1)) / (num_epochs * len(train_loader)) * 100
                }
                try:
                    progress_callback(prog)
                except Exception:
                    pass

            # Update progress bar
            progress_bar.set_postfix({'training_loss': f'{loss.item():.3f}'})
        
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_save_path = os.path.join('backend', 'model', 'car_crash_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model to {model_save_path}!")

        # Post epoch-level progress via callback
        if progress_callback is not None:
            try:
                progress_callback({
                    'is_training': True,
                    'epoch': epoch + 1,
                    'total_epochs': num_epochs,
                    'batch': len(train_loader),
                    'total_batches': len(train_loader),
                    'progress': (epoch + 1) / num_epochs * 100,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                })
            except Exception:
                pass
    
    # Clean up temporary files
    try:
        os.remove('train_temp.csv')
        os.remove('val_temp.csv')
    except Exception:
        pass

    # Save the final model state_dict
    print("Saving final model state_dict...")
    model.cpu()
    model_save_path = os.path.join('backend', 'model', 'car_crash_model.pth')
    torch.save(model.state_dict(), model_save_path)

    # Write a small JSON summary of the run so the frontend can read it
    try:
        summary = {
            'is_training': False,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'best_val_loss': best_val_loss
        }
        import json
        with open(os.path.join('backend', 'model', 'training_results.json'), 'w') as f:
            json.dump(summary, f)
    except Exception:
        pass

    # Final progress callback (training finished)
    if progress_callback is not None:
        try:
            progress_callback({'is_training': False, 'progress': 100})
        except Exception:
            pass

    print(f"Training completed. Model saved as {model_save_path}")

if __name__ == "__main__":
    train_model()