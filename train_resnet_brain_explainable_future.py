#!/usr/bin/env python3
"""
Explainable ResNet Brain Classifier (train_resnet_brain_explainable.py)
MONAI ResNet approach with interpretability methods for brain MRI classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import seaborn as sns

# MONAI imports
import monai
from monai.networks.nets import ResNet
from monai.utils import set_determinism

print(f"MONAI version: {monai.__version__}")
print(f"PyTorch version: {torch.__version__}")

set_determinism(seed=42)

class GradCAM:
    """Gradient-weighted Class Activation Mapping for 3D volumes"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for gradients and activations using modern PyTorch approach"""
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        # Find target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is not None:
            # Use register_full_backward_hook to avoid deprecation warning
            self.hooks.append(target_layer.register_full_backward_hook(backward_hook))
            self.hooks.append(target_layer.register_forward_hook(forward_hook))
        else:
            print(f"Warning: Target layer '{self.target_layer_name}' not found")
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate CAM for given input"""
        self.model.eval()
        
        # Ensure input requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Check if gradients and activations were captured
        if self.gradients is None or self.activations is None:
            print("Warning: Gradients or activations not captured properly")
            return np.zeros((64, 64, 64)), class_idx, 0.0
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients.view(gradients.size(0), -1), dim=1)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            if i < activations.shape[0]:
                cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), class_idx, output[0].softmax(0)[class_idx].item()
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class IntegratedGradients:
    """Integrated Gradients for 3D volumes"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_attributions(self, input_tensor, target_class=None, steps=50):
        """Generate integrated gradients attributions"""
        self.model.eval()
        
        # Clone input and ensure it requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps, device=input_tensor.device)
        
        attributions = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Clear any existing gradients
            if input_tensor.grad is not None:
                input_tensor.grad.zero_()
            
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated = interpolated.detach().requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Compute gradients
            class_score = output[0, target_class]
            class_score.backward()
            
            # Accumulate attributions
            if interpolated.grad is not None:
                attributions += interpolated.grad.detach()
        
        # Average and scale by input difference
        attributions = attributions / steps
        attributions = attributions * (input_tensor - baseline)
        
        return attributions[0].cpu().numpy(), target_class  # Remove batch dimension

class SaliencyMap:
    """Simple gradient-based saliency maps"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_saliency(self, input_tensor, target_class=None):
        """Generate saliency map"""
        self.model.eval()
        
        # Clone input and ensure it requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients
        if input_tensor.grad is not None:
            saliency = input_tensor.grad[0].abs()  # Remove batch dimension
        else:
            print("Warning: No gradients computed for saliency map")
            saliency = torch.zeros_like(input_tensor[0])
        
        return saliency.cpu().numpy(), target_class

class BrainDataset(Dataset):
    """Custom Dataset for brain MRI classification using ResNet"""
    
    def __init__(self, data_list, target_size=(128, 128, 128), augment=False):
        self.data_list = data_list
        self.target_size = target_size
        self.augment = augment
        
        print(f"Loading {len(data_list)} samples for ResNet...")
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

class ExplainableResNetBrainTrainer:
    """Explainable trainer class using MONAI ResNet for brain classification"""
    
    def __init__(self, data_dir, device='cuda', resnet_layers=[2, 2, 2, 2]):
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.resnet_layers = resnet_layers
        
        print(f"Using device: {self.device}")
        print(f"ResNet layers: {resnet_layers}")
        
        # Load data
        self.load_processed_data()
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Explainability tools
        self.grad_cam = None
        self.integrated_gradients = None
        self.saliency_map = None
        
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
        print("Creating ResNet datasets...")
        
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
    
    def initialize_model(self, learning_rate=1e-4, block_type='basic'):
        """Initialize ResNet model and explainability tools"""
        
        # Create ResNet model
        self.model = ResNet(
            block=block_type,  # 'basic' or 'bottleneck'
            layers=self.resnet_layers,
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2  # binary classification
        ).to(self.device)
        
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
        
        # Initialize explainability tools
        self._initialize_explainability_tools()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        layer_config = f"ResNet{sum(self.resnet_layers)*2+2}"  # Approximate naming
        print(f"{layer_config} initialized:")
        print(f"  - Block type: {block_type}")
        print(f"  - Layers: {self.resnet_layers}")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Explainability tools: Initialized")
    
    def _initialize_explainability_tools(self):
        """Initialize explainability tools"""
        # Find appropriate layer for GradCAM (last conv layer before classification)
        target_layer = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                target_layer = name
        
        if target_layer:
            self.grad_cam = GradCAM(self.model, target_layer)
            print(f"  - GradCAM target layer: {target_layer}")
        
        self.integrated_gradients = IntegratedGradients(self.model)
        self.saliency_map = SaliencyMap(self.model)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_labels = []
        all_probs = []
        
        progress_bar = tqdm(self.train_loader, desc="Training ResNet")
        
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
        """Train the ResNet model"""
        
        layer_config = f"ResNet{sum(self.resnet_layers)*2+2}"
        print(f"\nStarting {layer_config} training for {num_epochs} epochs...")
        
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
                    'resnet_layers': self.resnet_layers
                }, f'best_{layer_config.lower()}_brain_mapper.pth')
                
                print(f"New best model saved! Val AUC: {val_auc:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        
        print(f"\n{layer_config} training completed!")
        print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch+1})")
        print(f"Total training time: {training_time/60:.1f} minutes")
        
        return self.history
    
    def generate_explanations(self, sample_data, output_dir, num_samples=5):
        """Generate explanations for sample predictions"""
        
        output_dir = Path(output_dir)
        explanations_dir = output_dir / "explanations"
        explanations_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating explanations for {num_samples} samples...")
        
        # Create dataset for explanations
        if len(self.test_data) > 0:
            explain_data = self.test_data[:num_samples]
        else:
            explain_data = self.val_data[:num_samples]
        
        explain_dataset = BrainDataset(explain_data, target_size=self.target_size, augment=False)
        
        for i, sample_idx in enumerate(range(min(num_samples, len(explain_dataset)))):
            try:
                sample = explain_dataset[sample_idx]
                input_tensor = sample['image'].unsqueeze(0).to(self.device)
                true_label = sample['label'].item()
                subject_id = sample['subject_id']
                
                print(f"\nGenerating explanations for sample {i+1}/{num_samples}")
                print(f"Subject ID: {subject_id}, True label: {true_label}")
                
                # Get prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_label = output.argmax(dim=1).item()
                    confidence = pred_probs[0, pred_label].item()
                
                print(f"Predicted: {pred_label}, Confidence: {confidence:.4f}")
                
                # Generate different explanations
                explanations = {}
                
                # 1. GradCAM
                if self.grad_cam:
                    try:
                        cam, class_idx, score = self.grad_cam.generate_cam(input_tensor, pred_label)
                        explanations['gradcam'] = cam
                        print("✓ GradCAM generated")
                    except Exception as e:
                        print(f"✗ GradCAM failed: {e}")
                
                # 2. Integrated Gradients
                try:
                    ig_attr, class_idx = self.integrated_gradients.generate_attributions(
                        input_tensor, pred_label
                    )
                    explanations['integrated_gradients'] = ig_attr
                    print("✓ Integrated Gradients generated")
                except Exception as e:
                    print(f"✗ Integrated Gradients failed: {e}")
                
                # 3. Saliency Map
                try:
                    saliency, class_idx = self.saliency_map.generate_saliency(
                        input_tensor.clone(), pred_label
                    )
                    explanations['saliency'] = saliency
                    print("✓ Saliency Map generated")
                except Exception as e:
                    print(f"✗ Saliency Map failed: {e}")
                
                # Visualize explanations
                self._visualize_explanations(
                    sample['image'].cpu().numpy(),
                    explanations,
                    {
                        'subject_id': subject_id,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence
                    },
                    explanations_dir / f"explanation_{i+1}_{subject_id}.png"
                )
                
            except Exception as e:
                print(f"Error generating explanations for sample {i+1}: {e}")
                continue
        
        print(f"\nExplanations saved to: {explanations_dir}")
    
    def _visualize_explanations(self, original_image, explanations, metadata, save_path):
        """Visualize explanations for a single sample"""
        
        # Take middle slices for visualization
        h, w, d = original_image.shape[1:]
        mid_slice = d // 2
        mid_sagittal = w // 2
        
        # Number of explanation methods + original
        n_methods = len(explanations) + 1
        
        fig, axes = plt.subplots(2, n_methods, figsize=(4 * n_methods, 8))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        # Original image (top row, first column)
        original_slice = original_image[0, :, :, mid_slice]
        axes[0, 0].imshow(original_slice, cmap='gray')
        axes[0, 0].set_title('Original Image\n(Axial Slice)')
        axes[0, 0].axis('off')
        
        # Sagittal slice (bottom row, first column)
        sagittal_slice = original_image[0, :, mid_sagittal, :]
        axes[1, 0].imshow(sagittal_slice, cmap='gray')
        axes[1, 0].set_title('Original Image\n(Sagittal Slice)')
        axes[1, 0].axis('off')
        
        # Explanation methods
        col_idx = 1
        for method_name, explanation in explanations.items():
            
            if method_name == 'gradcam':
                # GradCAM - single channel
                explanation_slice = explanation[:, :, mid_slice]
                explanation_sagittal = explanation[:, mid_sagittal, :]
                
                # Overlay on original
                axes[0, col_idx].imshow(original_slice, cmap='gray', alpha=0.7)
                axes[0, col_idx].imshow(explanation_slice, cmap='jet', alpha=0.5)
                axes[0, col_idx].set_title(f'{method_name.upper()}\n(Axial Overlay)')
                
                axes[1, col_idx].imshow(sagittal_slice, cmap='gray', alpha=0.7)
                axes[1, col_idx].imshow(explanation_sagittal, cmap='jet', alpha=0.5)
                axes[1, col_idx].set_title(f'{method_name.upper()}\n(Sagittal Overlay)')
                
            else:
                # Other methods - take first channel if multi-channel
                if len(explanation.shape) == 4:
                    explanation = explanation[0]  # First channel
                
                explanation_slice = np.abs(explanation[:, :, mid_slice])
                explanation_sagittal = np.abs(explanation[:, mid_sagittal, :])
                
                # Normalize
                if explanation_slice.max() > 0:
                    explanation_slice = explanation_slice / explanation_slice.max()
                if explanation_sagittal.max() > 0:
                    explanation_sagittal = explanation_sagittal / explanation_sagittal.max()
                
                # Show as heatmap
                axes[0, col_idx].imshow(explanation_slice, cmap='hot')
                axes[0, col_idx].set_title(f'{method_name.replace("_", " ").title()}\n(Axial)')
                
                axes[1, col_idx].imshow(explanation_sagittal, cmap='hot')
                axes[1, col_idx].set_title(f'{method_name.replace("_", " ").title()}\n(Sagittal)')
            
            axes[0, col_idx].axis('off')
            axes[1, col_idx].axis('off')
            col_idx += 1
        
        # Add metadata as text
        metadata_text = f"""
Subject: {metadata['subject_id']}
True Label: {metadata['true_label']}
Predicted: {metadata['pred_label']}
Confidence: {metadata['confidence']:.3f}
Match: {'✓' if metadata['true_label'] == metadata['pred_label'] else '✗'}
        """
        
        plt.figtext(0.02, 0.02, metadata_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_importance(self, output_dir, num_samples=10):
        """Analyze which brain regions are most important for classification"""
        
        output_dir = Path(output_dir)
        analysis_dir = output_dir / "feature_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nAnalyzing feature importance across {num_samples} samples...")
        
        # Collect data
        if len(self.test_data) > 0:
            analyze_data = self.test_data[:num_samples]
        else:
            analyze_data = self.val_data[:num_samples]
        
        analyze_dataset = BrainDataset(analyze_data, target_size=self.target_size, augment=False)
        
        # Accumulate attribution maps
        all_attributions = {'case': [], 'control': []}
        all_predictions = {'case': [], 'control': []}
        all_confidences = {'case': [], 'control': []}
        prediction_stats = []
        
        for i in range(min(num_samples, len(analyze_dataset))):
            try:
                sample = analyze_dataset[i]
                input_tensor = sample['image'].unsqueeze(0).to(self.device)
                true_label = sample['label'].item()
                subject_id = sample['subject_id']
                
                # Get prediction with confidence
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred_probs = torch.softmax(output, dim=1)
                    pred_label = output.argmax(dim=1).item()
                    confidence = pred_probs[0, pred_label].item()
                
                # Generate integrated gradients (with proper detachment)
                ig_attr, _ = self.integrated_gradients.generate_attributions(
                    input_tensor, pred_label
                )
                
                # Store by class
                class_name = 'case' if true_label == 1 else 'control'
                all_attributions[class_name].append(ig_attr[0])  # First channel, already numpy
                all_predictions[class_name].append(pred_label)
                all_confidences[class_name].append(confidence)
                
                # Store prediction statistics
                prediction_stats.append({
                    'subject_id': subject_id,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'correct': true_label == pred_label,
                    'class_name': class_name
                })
                
                print(f"Processed sample {i+1}/{num_samples}: {subject_id} ({class_name})")
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Create comprehensive analysis
        self._create_population_analysis(all_attributions, all_predictions, all_confidences, 
                                       prediction_stats, analysis_dir)
        
        # Create average attribution maps
        self._create_average_attribution_maps(all_attributions, analysis_dir)
        
        # Analyze regional importance
        self._analyze_regional_importance(all_attributions, analysis_dir)
        
        print(f"Feature analysis saved to: {analysis_dir}")
    
    def _create_population_analysis(self, all_attributions, all_predictions, all_confidences, 
                                  prediction_stats, output_dir):
        """Create comprehensive population-level analysis"""
        
        # Convert prediction stats to DataFrame
        df_stats = pd.DataFrame(prediction_stats)
        
        if df_stats.empty:
            print("No data available for population analysis")
            return
        
        # Create a comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Prediction Accuracy by Class
        ax1 = plt.subplot(3, 4, 1)
        accuracy_by_class = df_stats.groupby('class_name')['correct'].agg(['mean', 'count']).reset_index()
        bars = ax1.bar(accuracy_by_class['class_name'], accuracy_by_class['mean'], 
                      color=['skyblue', 'lightcoral'])
        ax1.set_title('Prediction Accuracy by Class')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, accuracy_by_class['count'])):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'n={count}', ha='center', va='bottom')
        
        # 2. Confidence Distribution
        ax2 = plt.subplot(3, 4, 2)
        for class_name in df_stats['class_name'].unique():
            class_data = df_stats[df_stats['class_name'] == class_name]
            ax2.hist(class_data['confidence'], alpha=0.7, label=class_name, bins=10)
        ax2.set_title('Prediction Confidence Distribution')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.legend()
        
        # 3. Confidence vs Accuracy
        ax3 = plt.subplot(3, 4, 3)
        colors = ['red' if not correct else 'green' for correct in df_stats['correct']]
        scatter = ax3.scatter(df_stats['confidence'], df_stats['true_label'], 
                            c=colors, alpha=0.6)
        ax3.set_title('Confidence vs True Label')
        ax3.set_xlabel('Prediction Confidence')
        ax3.set_ylabel('True Label')
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Control', 'Case'])
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Correct'),
                          Patch(facecolor='red', label='Incorrect')]
        ax3.legend(handles=legend_elements)
        
        # 4. Overall Performance Metrics
        ax4 = plt.subplot(3, 4, 4)
        ax4.axis('off')
        
        overall_accuracy = df_stats['correct'].mean()
        total_samples = len(df_stats)
        correct_predictions = df_stats['correct'].sum()
        avg_confidence = df_stats['confidence'].mean()
        
        # Calculate per-class metrics
        case_accuracy = df_stats[df_stats['class_name'] == 'case']['correct'].mean() if 'case' in df_stats['class_name'].values else 0
        control_accuracy = df_stats[df_stats['class_name'] == 'control']['correct'].mean() if 'control' in df_stats['class_name'].values else 0
        
        metrics_text = f"""
PERFORMANCE SUMMARY

Overall Accuracy: {overall_accuracy:.3f}
Total Samples: {total_samples}
Correct Predictions: {correct_predictions}
Average Confidence: {avg_confidence:.3f}

CLASS-SPECIFIC ACCURACY:
Case Accuracy: {case_accuracy:.3f}
Control Accuracy: {control_accuracy:.3f}

SAMPLE DISTRIBUTION:
Cases: {sum(df_stats['class_name'] == 'case')}
Controls: {sum(df_stats['class_name'] == 'control')}
        """
        
        ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        # 5-8. Average Attribution Heatmaps for each class
        for i, class_name in enumerate(['case', 'control']):
            if class_name in all_attributions and all_attributions[class_name]:
                avg_attr = np.mean(all_attributions[class_name], axis=0)
                avg_attr = np.abs(avg_attr)
                if avg_attr.max() > 0:
                    avg_attr = avg_attr / avg_attr.max()
                
                h, w, d = avg_attr.shape
                
                # Axial view
                ax_axial = plt.subplot(3, 4, 5 + i*2)
                axial_slice = avg_attr[:, :, d//2]
                im1 = ax_axial.imshow(axial_slice, cmap='hot', aspect='equal')
                ax_axial.set_title(f'{class_name.title()} - Axial View\n(n={len(all_attributions[class_name])})')
                ax_axial.axis('off')
                plt.colorbar(im1, ax=ax_axial, fraction=0.046, pad=0.04)
                
                # Sagittal view
                ax_sagittal = plt.subplot(3, 4, 6 + i*2)
                sagittal_slice = avg_attr[:, w//2, :]
                im2 = ax_sagittal.imshow(sagittal_slice, cmap='hot', aspect='equal')
                ax_sagittal.set_title(f'{class_name.title()} - Sagittal View')
                ax_sagittal.axis('off')
                plt.colorbar(im2, ax=ax_sagittal, fraction=0.046, pad=0.04)
        
        # 9. Subject-wise Confidence Plot
        ax9 = plt.subplot(3, 4, 9)
        x_pos = range(len(df_stats))
        colors = ['lightcoral' if class_name == 'case' else 'skyblue' 
                 for class_name in df_stats['class_name']]
        bars = ax9.bar(x_pos, df_stats['confidence'], color=colors, alpha=0.7)
        
        # Mark incorrect predictions
        for i, (conf, correct) in enumerate(zip(df_stats['confidence'], df_stats['correct'])):
            if not correct:
                ax9.bar(i, conf, color='red', alpha=0.8)
        
        ax9.set_title('Per-Subject Confidence\n(Red = Incorrect Predictions)')
        ax9.set_xlabel('Subject Index')
        ax9.set_ylabel('Confidence')
        ax9.set_ylim(0, 1)
        
        # 10. Prediction Matrix/Confusion-style visualization
        ax10 = plt.subplot(3, 4, 10)
        
        # Create confusion matrix data
        confusion_data = pd.crosstab(df_stats['true_label'], df_stats['pred_label'], 
                                   normalize='index')
        
        if not confusion_data.empty:
            im = ax10.imshow(confusion_data.values, cmap='Blues', aspect='equal')
            ax10.set_title('Normalized Confusion Matrix')
            ax10.set_xlabel('Predicted Label')
            ax10.set_ylabel('True Label')
            ax10.set_xticks([0, 1])
            ax10.set_yticks([0, 1])
            ax10.set_xticklabels(['Control', 'Case'])
            ax10.set_yticklabels(['Control', 'Case'])
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    if i < len(confusion_data.values) and j < len(confusion_data.values[0]):
                        text = ax10.text(j, i, f'{confusion_data.values[i, j]:.2f}',
                                       ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax10, fraction=0.046, pad=0.04)
        
        # 11. Attribution Intensity Distribution
        ax11 = plt.subplot(3, 4, 11)
        for class_name in all_attributions:
            if all_attributions[class_name]:
                # Calculate mean attribution intensity per subject
                intensities = [np.mean(np.abs(attr)) for attr in all_attributions[class_name]]
                ax11.hist(intensities, alpha=0.7, label=f'{class_name} (n={len(intensities)})', bins=8)
        
        ax11.set_title('Attribution Intensity Distribution')
        ax11.set_xlabel('Mean Attribution Intensity')
        ax11.set_ylabel('Count')
        ax11.legend()
        
        # 12. Model Architecture Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        layer_config = f"ResNet{sum(self.resnet_layers)*2+2}"
        total_params = sum(p.numel() for p in self.model.parameters())
        
        model_info = f"""
MODEL ARCHITECTURE

Architecture: {layer_config}
Layers: {self.resnet_layers}
Total Parameters: {total_params:,}
Device: {self.device}

ANALYSIS DETAILS

Explanation Method: Integrated Gradients
Attribution Steps: 50
Target Size: {self.target_size}
Analysis Samples: {len(df_stats)}
        """
        
        ax12.text(0.05, 0.95, model_info, transform=ax12.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        
        plt.suptitle('Population-Level Brain Classification Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'population_analysis_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed statistics
        df_stats.to_csv(output_dir / 'prediction_statistics.csv', index=False)
        
        print("✓ Comprehensive population analysis created")
        print(f"  - Overall accuracy: {overall_accuracy:.3f}")
        print(f"  - Total samples analyzed: {total_samples}")
        print(f"  - Results saved to: population_analysis_comprehensive.png")
    
    def _create_average_attribution_maps(self, all_attributions, output_dir):
        """Create average attribution maps for each class"""
        
        for class_name, attributions in all_attributions.items():
            if not attributions:
                continue
            
            # Average attributions
            avg_attribution = np.mean(attributions, axis=0)
            
            # Take absolute values and normalize
            avg_attribution = np.abs(avg_attribution)
            if avg_attribution.max() > 0:
                avg_attribution = avg_attribution / avg_attribution.max()
            
            # Visualize different slices
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            h, w, d = avg_attribution.shape
            
            # Axial slices
            for i, slice_idx in enumerate([d//4, d//2, 3*d//4]):
                axes[0, i].imshow(avg_attribution[:, :, slice_idx], cmap='hot')
                axes[0, i].set_title(f'Axial Slice {slice_idx}')
                axes[0, i].axis('off')
            
            # Sagittal slices
            for i, slice_idx in enumerate([w//4, w//2, 3*w//4]):
                axes[1, i].imshow(avg_attribution[:, slice_idx, :], cmap='hot')
                axes[1, i].set_title(f'Sagittal Slice {slice_idx}')
                axes[1, i].axis('off')
            
            plt.suptitle(f'Average Feature Importance - {class_name.title()} Class\n'
                        f'(n={len(attributions)} samples)', fontsize=14)
            plt.tight_layout()
            plt.savefig(output_dir / f'avg_attribution_{class_name}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _analyze_regional_importance(self, all_attributions, output_dir):
        """Analyze importance by brain regions"""
        
        # Simple regional analysis by dividing brain into regions
        results = []
        
        for class_name, attributions in all_attributions.items():
            if not attributions:
                continue
            
            for attr in attributions:
                h, w, d = attr.shape
                
                # Define brain regions (simplified)
                regions = {
                    'anterior': attr[:, :, :d//3],
                    'middle': attr[:, :, d//3:2*d//3],
                    'posterior': attr[:, :, 2*d//3:],
                    'left': attr[:, :w//2, :],
                    'right': attr[:, w//2:, :],
                    'superior': attr[:h//2, :, :],
                    'inferior': attr[h//2:, :, :]
                }
                
                # Calculate average importance for each region
                for region_name, region_data in regions.items():
                    avg_importance = np.mean(np.abs(region_data))
                    results.append({
                        'class': class_name,
                        'region': region_name,
                        'importance': avg_importance
                    })
        
        # Create DataFrame and plot
        df_results = pd.DataFrame(results)
        
        if not df_results.empty:
            # Group by class and region
            grouped = df_results.groupby(['class', 'region'])['importance'].mean().reset_index()
            
            # Pivot for plotting
            pivot_df = grouped.pivot(index='region', columns='class', values='importance')
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot_df.plot(kind='bar', ax=ax)
            ax.set_title('Average Feature Importance by Brain Region')
            ax.set_xlabel('Brain Region')
            ax.set_ylabel('Average Attribution')
            ax.legend(title='Class')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_dir / 'regional_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save data
            pivot_df.to_csv(output_dir / 'regional_importance.csv')
    
    def evaluate_model(self):
        """Evaluate model on test set"""
        if len(self.test_data) == 0:
            print("No test data available")
            return None
        
        layer_config = f"ResNet{sum(self.resnet_layers)*2+2}"
        print(f"\nEvaluating {layer_config} on {len(self.test_data)} test subjects...")
        
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
        
        print(f"{layer_config} Test Results:")
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
        
        layer_config = f"ResNet{sum(self.resnet_layers)*2+2}"
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title(f'{layer_config} Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        axes[0, 1].set_title(f'{layer_config} Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        axes[1, 0].plot(epochs, self.history['train_auc'], 'b-', label='Train AUC')
        axes[1, 0].plot(epochs, self.history['val_auc'], 'r-', label='Val AUC')
        axes[1, 0].set_title(f'{layer_config} Training AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Model info
        axes[1, 1].text(0.1, 0.9, f'{layer_config} Configuration:', 
                       transform=axes[1, 1].transAxes, fontweight='bold', fontsize=12)
        
        info_text = f"""
Architecture: {layer_config}
Layers: {self.resnet_layers}
Spatial Dims: 3D
Input Channels: 1
Output Classes: 2
Optimizer: Adam
Loss: CrossEntropy
Data: {len(self.train_data)} train, {len(self.val_data)} val

Explainability Methods:
• GradCAM
• Integrated Gradients  
• Saliency Maps
• Feature Analysis
        """
        
        axes[1, 1].text(0.1, 0.1, info_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='bottom', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{layer_config.lower()}_explainable_training_history.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def cleanup_explainability_tools(self):
        """Cleanup explainability tools"""
        if self.grad_cam:
            self.grad_cam.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Explainable ResNet Brain Classification")
    
    parser.add_argument('--data_dir', type=str, default='./my_processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='./explainable_resnet_results',
                       help='Output directory for results')
    parser.add_argument('--resnet_layers', nargs=4, type=int, default=[2, 2, 2, 2],
                       help='ResNet layer configuration (4 integers)')
    parser.add_argument('--block_type', type=str, default='basic',
                       choices=['basic', 'bottleneck'],
                       help='ResNet block type')
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
    parser.add_argument('--generate_explanations', action='store_true',
                       help='Generate explanations for sample predictions')
    parser.add_argument('--num_explain_samples', type=int, default=5,
                       help='Number of samples to explain')
    parser.add_argument('--feature_analysis', action='store_true',
                       help='Perform feature importance analysis')
    parser.add_argument('--num_analysis_samples', type=int, default=10,
                       help='Number of samples for feature analysis')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to pre-trained model to load')
    
    args = parser.parse_args()
    
    layer_config = f"ResNet{sum(args.resnet_layers)*2+2}"
    
    print("=== Explainable MONAI ResNet Brain Classification ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Architecture: {layer_config}")
    print(f"Layers: {args.resnet_layers}")
    print(f"Block type: {args.block_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Explainability: Enabled")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = ExplainableResNetBrainTrainer(
            args.data_dir, 
            device=args.device, 
            resnet_layers=args.resnet_layers
        )
        
        # Create data loaders
        trainer.create_data_loaders(batch_size=args.batch_size)
        
        # Initialize model
        trainer.initialize_model(learning_rate=args.learning_rate, block_type=args.block_type)
        
        # Load pre-trained model if specified
        if args.load_model and Path(args.load_model).exists():
            print(f"Loading pre-trained model: {args.load_model}")
            checkpoint = torch.load(args.load_model, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully")
        else:
            # Train model
            history = trainer.train_model(num_epochs=args.num_epochs)
            
            # Plot training history
            trainer.plot_training_history(args.output_dir)
        
        # Evaluate on test set if requested
        if args.evaluate and len(trainer.test_data) > 0:
            test_results = trainer.evaluate_model()
        
        # Generate explanations if requested
        if args.generate_explanations:
            trainer.generate_explanations(
                trainer.test_data if len(trainer.test_data) > 0 else trainer.val_data,
                args.output_dir,
                num_samples=args.num_explain_samples
            )
        
        # Feature analysis if requested
        if args.feature_analysis:
            trainer.analyze_feature_importance(
                args.output_dir,
                num_samples=args.num_analysis_samples
            )
        
        # Cleanup
        trainer.cleanup_explainability_tools()
        
        print(f"\n{layer_config} explainable training completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Model saved as: best_{layer_config.lower()}_brain_mapper.pth")
        
        if args.generate_explanations:
            print(f"Explanations saved to: {args.output_dir}/explanations/")
        
        if args.feature_analysis:
            print(f"Feature analysis saved to: {args.output_dir}/feature_analysis/")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
