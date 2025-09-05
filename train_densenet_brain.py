#!/usr/bin/env python3
"""
DenseNet Brain Classifier (train_densenet_brain.py)
Standard MONAI DenseNet approach for brain MRI classification
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import json
import pandas as pd
from pathlib import Path
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
import time
import random

# MONAI imports
import monai
from monai.networks.nets import DenseNet121, DenseNet169, DenseNet201
from monai.utils import set_determinism

print(f"MONAI version: {monai.__version__}")
print(f"PyTorch version: {torch.__version__}")

set_determinism(seed=42)

class BrainDataset(Dataset):
    """Custom Dataset for brain MRI classification using DenseNet"""
    
    def __init__(self, data_list, target_size=(128, 128, 128), augment=False):
        self.data_list = data_list
        self.target_size = target_size
        self.augment = augment
        
        print(f"Loading {len(data_list)} samples for DenseNet...")
        self.loaded_data = []
        
        for item in tqdm(data_list, desc="Loading data"):
            try:
                # Load numpy array
                image_data = np.load(item['image'])
                
                # Ensure float32
                if image_data.dtype != np.float32:
                    image_data = image_data.astype(np.float32)
                
                # Ensure proper shape: (C, H, W, D)
                if len(image_data.shape) == 3:  # (H, W, D) -> (1, H, W, D)
                    image_data = image_data[np.newaxis, ...]
                elif len(image_data.shape) == 4 and image_data.shape[0] != 1:
                    image_data = image_data[0:1, ...]
                
                # Resize if needed
                if image_data.shape[1:] != target_size:
                    image_data = self.resize_volume(image_data, target_size)
                
                # Normalize to [0, 1]
                if image_data.max() > 1.0:
                    image_data = image_data / image_data.max()
                
                self.loaded_data.append({
                    'image': image_data,
                    'label': item['label'],
                    'subject_id': item['subject_id']
                })
                
            except Exception as e:
                print(f"Error loading {item['image']}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.loaded_data)} samples")
    
    def resize_volume(self, image_data, target_size):
        """Simple resize using scipy if available"""
        try:
            from scipy.ndimage import zoom
            
            current_size = image_data.shape[1:]
            zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
            
            resized_channels = []
            for c in range(image_data.shape[0]):
                resized_channel = zoom(image_data[c], zoom_factors, order=1)
                resized_channels.append(resized_channel)
            
            return np.stack(resized_channels, axis=0)
            
        except ImportError:
            print("Warning: scipy not available, using original size")
            return image_data
    
    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        item = self.loaded_data[idx]
        
        image = item['image'].copy()
        label = item['label']
        subject_id = item['subject_id']
        
        # Simple augmentation
        if self.augment:
            image = self.apply_augmentation(image)
        
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'subject_id': subject_id
        }
    
    def apply_augmentation(self, image):
        """Simple augmentation"""
        # Random flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
        if random.random() > 0.5:
            image = np.flip(image, axis=3).copy()
        
        # Add noise
        if random.random() > 0.8:
            noise = np.random.normal(0, 0.01, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0, 1)
        
        return image

class DenseNetBrainTrainer:
    """Trainer class using MONAI DenseNet for brain classification"""
    
    def __init__(self, data_dir, device='cuda', densenet_version='121'):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.densenet_version = densenet_version
        
        print(f"Using device: {self.device}")
        print(f"DenseNet version: {densenet_version}")
        
        # Load data
        self.load_processed_data()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': []
        }
    
    def load_processed_data(self):
        """Load processed data splits"""
        splits_file = self.data_dir / "data_splits.json"
        if not splits_file.exists():
            raise FileNotFoundError(f"Data splits file not found: {splits_file}")
        
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        
        self.train_data = splits_data['train_data']
        self.val_data = splits_data['val_data']
        self.test_data = splits_data['test_data']
        self.target_size = tuple(splits_data['target_size'])
        
        print(f"Loaded data splits:")
        print(f"  - Train: {len(self.train_data)} subjects")
        print(f"  - Validation: {len(self.val_data)} subjects")
        print(f"  - Test: {len(self.test_data)} subjects")
        print(f"  - Target size: {self.target_size}")
    
    def create_data_loaders(self, batch_size=4):
        """Create data loaders"""
        print("Creating DenseNet datasets...")
        
        train_dataset = BrainDataset(
            self.train_data, 
            target_size=self.target_size, 
            augment=True
        )
        val_dataset = BrainDataset(
            self.val_data, 
            target_size=self.target_size, 
            augment=False
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"Data loaders created:")
        print(f"  - Train batches: {len(self.train_loader)}")
        print(f"  - Val batches: {len(self.val_loader)}")
        print(f"  - Batch size: {batch_size}")
    
    def initialize_model(self, learning_rate=1e-4):
        """Initialize DenseNet model"""
        
        # Select DenseNet version
        if self.densenet_version == '121':
            self.model = DenseNet121(
                spatial_dims=3,
                in_channels=1,
                out_channels=2  # binary classification
            ).to(self.device)
        elif self.densenet_version == '169':
            self.model = DenseNet169(
                spatial_dims=3,
                in_channels=1,
                out_channels=2
            ).to(self.device)
        elif self.densenet_version == '201':
            self.model = DenseNet201(
                spatial_dims=3,
                in_channels=1,
                out_channels=2
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported DenseNet version: {self.densenet_version}")
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"DenseNet{self.densenet_version} initialized:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Learning rate: {learning_rate}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_labels = []
        all_probs = []
        
        progress_bar = tqdm(self.train_loader, desc="Training DenseNet")
        
        for batch_data in progress_bar:
            inputs = batch_data["image"].to(self.device)
            labels = batch_data["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            if torch.isnan(outputs).any():
                continue
            
            loss = self.criterion(outputs, labels)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # For AUC
            probs = torch.softmax(outputs, dim=1)[:, 1]
            if not torch.isnan(probs).any():
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.detach().cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_correct/train_total:.4f}'
            })
        
        avg_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        accuracy = train_correct / train_total if train_total > 0 else 0
        
        if len(all_probs) > 0 and len(np.unique(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5
        
        return avg_loss, accuracy, auc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc="Validating"):
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                outputs = self.model(inputs)
                
                if torch.isnan(outputs).any():
                    continue
                
                loss = self.criterion(outputs, labels)
                
                if torch.isnan(loss):
                    continue
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                if not torch.isnan(probs).any():
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.detach().cpu().numpy())
        
        avg_loss = val_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        accuracy = val_correct / val_total if val_total > 0 else 0
        
        if len(all_probs) > 0 and len(np.unique(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5
        
        return avg_loss, accuracy, auc
    
    def train_model(self, num_epochs=50):
        """Train the DenseNet model"""
        
        print(f"\nStarting DenseNet{self.densenet_version} training for {num_epochs} epochs...")
        
        best_val_auc = 0.0
        best_epoch = 0
        patience = 15
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            train_loss, train_acc, train_auc = self.train_epoch()
            val_loss, val_acc, val_auc = self.validate_epoch()
            
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'densenet_version': self.densenet_version
                }, f'best_densenet{self.densenet_version}_brain_mapper.pth')
                
                print(f"New best model saved! Val AUC: {val_auc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        
        print(f"\nDenseNet{self.densenet_version} training completed!")
        print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
        print(f"Total training time: {training_time/60:.1f} minutes")
        
        return self.history
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        if len(self.test_data) == 0:
            print("No test data available")
            return None
        
        print(f"\nEvaluating DenseNet{self.densenet_version} on {len(self.test_data)} test subjects...")
        
        test_dataset = BrainDataset(
            self.test_data,
            target_size=self.target_size,
            augment=False
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        self.model.eval()
        all_labels = []
        all_probs = []
        all_predictions = []
        
        with torch.no_grad():
            for batch_data in tqdm(test_loader, desc="Testing"):
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)
                
                outputs = self.model(inputs)
                
                if torch.isnan(outputs).any():
                    continue
                
                probs = torch.softmax(outputs, dim=1)
                if torch.isnan(probs).any():
                    continue
                    
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        if len(all_labels) == 0:
            print("Error: No valid predictions")
            return None
        
        accuracy = accuracy_score(all_labels, all_predictions)
        
        if len(np.unique(all_labels)) > 1:
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float('nan')
        else:
            auc = float('nan')
        
        print(f"DenseNet{self.densenet_version} Test Results:")
        print(f"  - Accuracy: {accuracy:.4f}")
        if not np.isnan(auc):
            print(f"  - AUC: {auc:.4f}")
        
        try:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_predictions, 
                                      target_names=['Control', 'Case']))
        except Exception as e:
            print(f"Could not generate classification report: {e}")
        
        return {
            'accuracy': accuracy,
            'auc': auc if not np.isnan(auc) else None,
            'labels': all_labels,
            'probabilities': all_probs,
            'predictions': all_predictions
        }
    
    def plot_training_history(self, output_dir):
        """Plot training history"""
        if not self.history['train_loss']:
            print("No training history available")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title(f'DenseNet{self.densenet_version} Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title(f'DenseNet{self.densenet_version} Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(epochs, self.history['train_auc'], 'b-', label='Train AUC')
        axes[1, 0].plot(epochs, self.history['val_auc'], 'r-', label='Val AUC')
        axes[1, 0].set_title(f'DenseNet{self.densenet_version} Training AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Model info
        axes[1, 1].text(0.1, 0.9, f'DenseNet{self.densenet_version} Configuration:', 
                       transform=axes[1, 1].transAxes, fontweight='bold', fontsize=12)
        
        info_text = f"""
Architecture: DenseNet{self.densenet_version}
Spatial Dims: 3D
Input Channels: 1
Output Classes: 2
Optimizer: Adam
Loss: CrossEntropy
Data: {len(self.train_data)} train, {len(self.val_data)} val
        """
        
        axes[1, 1].text(0.1, 0.1, info_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='bottom', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"densenet{self.densenet_version}_training_history.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="DenseNet Brain Classification")
    
    parser.add_argument('--data_dir', type=str, default='./my_processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='./densenet_results',
                       help='Output directory for results')
    parser.add_argument('--densenet_version', type=str, default='121', 
                       choices=['121', '169', '201'],
                       help='DenseNet version to use')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on test set')
    
    args = parser.parse_args()
    
    print("=== MONAI DenseNet Brain Classification ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"DenseNet version: {args.densenet_version}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = DenseNetBrainTrainer(
            args.data_dir, 
            device=args.device, 
            densenet_version=args.densenet_version
        )
        
        # Create data loaders
        trainer.create_data_loaders(batch_size=args.batch_size)
        
        # Initialize model
        trainer.initialize_model(learning_rate=args.learning_rate)
        
        # Train model
        history = trainer.train_model(num_epochs=args.num_epochs)
        
        # Plot training history
        trainer.plot_training_history(args.output_dir)
        
        # Evaluate on test set if requested
        if args.evaluate and len(trainer.test_data) > 0:
            test_results = trainer.evaluate_model()
        
        print(f"\nDenseNet{args.densenet_version} training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Model saved as: best_densenet{args.densenet_version}_brain_mapper.pth")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
