#!/usr/bin/env python3
"""
Brain MRI Data Preprocessor with updated MONAI imports
Handles DICOM and NIfTI files for brain mapping model training
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pydicom
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from monai.transforms import (
        Compose, LoadImage, EnsureChannelFirst, Spacing, Orientation, 
        ScaleIntensity, Resize, ToTensor, RandRotate90, RandFlip,
        RandGaussianNoise, RandAdjustContrast
    )
    from monai.data import Dataset, DataLoader
    from monai.utils import set_determinism
    MONAI_AVAILABLE = True
except ImportError as e:
    print(f"MONAI not available: {e}")
    print("Please install MONAI: pip install monai[all]")
    MONAI_AVAILABLE = False

class BrainDataProcessor:
    def __init__(self, data_dir, target_size=(128, 128, 128), target_spacing=(1.0, 1.0, 1.0)):
        """
        Initialize the brain data processor
        
        Args:
            data_dir (str): Directory containing brain MRI data
            target_size (tuple): Target size for resized images
            target_spacing (tuple): Target spacing for resampling
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.data_info = []
        self.processed_data = []
        
        if not MONAI_AVAILABLE:
            raise ImportError("MONAI is required but not available")
        
        # Set determinism for reproducibility
        set_determinism(seed=42)
        
        # Define preprocessing transforms
        self.preprocessing_transforms = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),  # Updated from AddChannel
            Spacing(pixdim=self.target_spacing, mode="bilinear"),
            Orientation(axcodes="RAS"),
            ScaleIntensity(minv=0.0, maxv=1.0),
            Resize(spatial_size=self.target_size, mode="trilinear"),
            ToTensor()
        ])
        
        # Define augmentation transforms for training
        self.augmentation_transforms = Compose([
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            RandFlip(prob=0.5, spatial_axis=0),
            RandGaussianNoise(prob=0.1, std=0.01),
            RandAdjustContrast(prob=0.3, gamma=(0.8, 1.2))
        ])

    def scan_data_directory(self):
        """Scan the data directory and catalog available files"""
        print(f"Scanning directory: {self.data_dir}")
        
        supported_extensions = ['.nii', '.nii.gz', '.dcm']
        file_count = 0
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                file_path = Path(root) / file
                
                if any(str(file_path).lower().endswith(ext) for ext in supported_extensions):
                    try:
                        # Basic file info
                        file_info = {
                            'path': str(file_path),
                            'filename': file_path.name,
                            'size_mb': file_path.stat().st_size / (1024 * 1024),
                            'type': self._determine_file_type(file_path)
                        }
                        
                        # Try to get image dimensions
                        try:
                            if file_path.suffix.lower() in ['.nii', '.gz']:
                                img = nib.load(str(file_path))
                                file_info['shape'] = img.shape
                                file_info['spacing'] = img.header.get_zooms()
                            elif file_path.suffix.lower() == '.dcm':
                                dcm = pydicom.dcmread(str(file_path))
                                file_info['shape'] = (dcm.Rows, dcm.Columns)
                                
                        except Exception as e:
                            file_info['error'] = str(e)
                            
                        self.data_info.append(file_info)
                        file_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        print(f"Found {file_count} medical imaging files")
        return self.data_info

    def _determine_file_type(self, file_path):
        """Determine the type of medical imaging file"""
        suffix = file_path.suffix.lower()
        if suffix in ['.nii', '.gz']:
            return 'nifti'
        elif suffix == '.dcm':
            return 'dicom'
        else:
            return 'unknown'

    def preprocess_data(self, max_files=None):
        """
        Preprocess the medical imaging data
        
        Args:
            max_files (int): Maximum number of files to process (for testing)
        """
        if not self.data_info:
            print("No data found. Run scan_data_directory() first.")
            return
        
        files_to_process = self.data_info[:max_files] if max_files else self.data_info
        print(f"Preprocessing {len(files_to_process)} files...")
        
        successful_count = 0
        for i, file_info in enumerate(files_to_process):
            try:
                print(f"Processing {i+1}/{len(files_to_process)}: {file_info['filename']}")
                
                # Apply preprocessing transforms
                preprocessed = self.preprocessing_transforms(file_info['path'])
                
                processed_info = {
                    'original_path': file_info['path'],
                    'filename': file_info['filename'],
                    'processed_shape': preprocessed.shape,
                    'processed_data': preprocessed,
                    'original_info': file_info
                }
                
                self.processed_data.append(processed_info)
                successful_count += 1
                
            except Exception as e:
                print(f"Error preprocessing {file_info['filename']}: {e}")
        
        print(f"Successfully preprocessed {successful_count} files")
        return self.processed_data

    def analyze_data_distribution(self):
        """Analyze the distribution of processed data"""
        if not self.processed_data:
            print("No processed data available.")
            return
        
        print("\n" + "="*50)
        print("DATA ANALYSIS SUMMARY")
        print("="*50)
        
        # Basic statistics
        total_files = len(self.processed_data)
        print(f"Total processed files: {total_files}")
        
        # Shape analysis
        shapes = [data['processed_shape'] for data in self.processed_data]
        unique_shapes = list(set(shapes))
        print(f"Unique shapes: {unique_shapes}")
        
        # Intensity analysis
        intensities = []
        for data in self.processed_data:
            tensor_data = data['processed_data']
            intensities.extend([
                float(tensor_data.min()),
                float(tensor_data.max()),
                float(tensor_data.mean()),
                float(tensor_data.std())
            ])
        
        print(f"Intensity range: [{min(intensities):.3f}, {max(intensities):.3f}]")
        print(f"Average intensity: {np.mean(intensities):.3f} ± {np.std(intensities):.3f}")
        
        # File type distribution
        file_types = {}
        for data in self.processed_data:
            file_type = data['original_info']['type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        print("\nFile type distribution:")
        for file_type, count in file_types.items():
            print(f"  {file_type}: {count} files")

    def create_sample_batches(self, output_dir, batch_size=8):
        """Create sample data batches for model training"""
        if not self.processed_data:
            print("No processed data available.")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating sample batches with batch_size={batch_size}")
        
        # Create batches
        num_batches = len(self.processed_data) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_data = []
            batch_info = []
            
            for data in self.processed_data[start_idx:end_idx]:
                batch_data.append(data['processed_data'])
                batch_info.append({
                    'filename': data['filename'],
                    'original_path': data['original_path'],
                    'shape': data['processed_shape']
                })
            
            # Stack tensors
            try:
                batch_tensor = np.stack(batch_data, axis=0)
                
                # Save batch
                batch_file = output_dir / f"batch_{batch_idx:03d}.npy"
                np.save(batch_file, batch_tensor)
                
                # Save batch info
                info_file = output_dir / f"batch_{batch_idx:03d}_info.json"
                with open(info_file, 'w') as f:
                    json.dump(batch_info, f, indent=2)
                
                print(f"Saved batch {batch_idx}: {batch_file}")
                
            except Exception as e:
                print(f"Error creating batch {batch_idx}: {e}")
        
        print(f"Created {num_batches} batches in {output_dir}")

    def save_processed_data(self, output_dir):
        """Save all processed data and metadata"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processing metadata
        metadata = {
            'total_files': len(self.processed_data),
            'target_size': self.target_size,
            'target_spacing': self.target_spacing,
            'preprocessing_info': 'Applied LoadImage, EnsureChannelFirst, Spacing, Orientation, ScaleIntensity, Resize'
        }
        
        metadata_file = output_dir / 'processing_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save individual processed files
        processed_dir = output_dir / 'processed_files'
        processed_dir.mkdir(exist_ok=True)
        
        for i, data in enumerate(self.processed_data):
            filename = f"processed_{i:03d}_{Path(data['filename']).stem}.npy"
            file_path = processed_dir / filename
            
            # Convert tensor to numpy if needed
            tensor_data = data['processed_data']
            if hasattr(tensor_data, 'numpy'):
                numpy_data = tensor_data.numpy()
            else:
                numpy_data = np.array(tensor_data)
            
            np.save(file_path, numpy_data)
        
        print(f"Saved processed data to: {output_dir}")
        return str(output_dir)

    def visualize_preprocessing(self, output_dir):
        """Create visualizations of the preprocessing results"""
        if not self.processed_data:
            print("No processed data to visualize.")
            return
        
        output_dir = Path(output_dir)
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize a few sample slices
        num_samples = min(4, len(self.processed_data))
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
        if num_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(num_samples):
            data = self.processed_data[i]['processed_data']
            
            # Convert to numpy if needed
            if hasattr(data, 'numpy'):
                numpy_data = data.numpy()
            else:
                numpy_data = np.array(data)
            
            # Remove channel dimension if present
            if len(numpy_data.shape) == 4:
                numpy_data = numpy_data[0]  # Remove batch/channel dim
            
            # Get middle slices
            if len(numpy_data.shape) == 3:
                mid_axial = numpy_data.shape[2] // 2
                mid_sagittal = numpy_data.shape[0] // 2
                
                # Axial slice
                axes[0, i].imshow(numpy_data[:, :, mid_axial], cmap='gray')
                axes[0, i].set_title(f'Axial - {Path(self.processed_data[i]["filename"]).stem}')
                axes[0, i].axis('off')
                
                # Sagittal slice
                axes[1, i].imshow(numpy_data[mid_sagittal, :, :], cmap='gray')
                axes[1, i].set_title(f'Sagittal - {Path(self.processed_data[i]["filename"]).stem}')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'preprocessing_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create intensity distribution plot
        plt.figure(figsize=(12, 6))
        
        all_intensities = []
        for data in self.processed_data:
            tensor_data = data['processed_data']
            if hasattr(tensor_data, 'numpy'):
                numpy_data = tensor_data.numpy()
            else:
                numpy_data = np.array(tensor_data)
            all_intensities.extend(numpy_data.flatten())
        
        plt.hist(all_intensities, bins=100, alpha=0.7, density=True)
        plt.xlabel('Intensity Value')
        plt.ylabel('Density')
        plt.title('Intensity Distribution After Preprocessing')
        plt.grid(True, alpha=0.3)
        plt.savefig(viz_dir / 'intensity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {viz_dir}")

def main():
    parser = argparse.ArgumentParser(description='Process brain MRI data for model training')
    parser.add_argument('data_dir', help='Directory containing brain MRI data')
    parser.add_argument('--output-dir', default='./processed_brain_data', 
                       help='Output directory for processed data')
    parser.add_argument('--target-size', nargs=3, type=int, default=[128, 128, 128],
                       help='Target size for resizing (default: 128 128 128)')
    parser.add_argument('--target-spacing', nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help='Target spacing for resampling (default: 1.0 1.0 1.0)')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for creating sample batches')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of preprocessing results')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = BrainDataProcessor(
            data_dir=args.data_dir,
            target_size=tuple(args.target_size),
            target_spacing=tuple(args.target_spacing)
        )
        
        print("Brain MRI Data Preprocessor")
        print("="*50)
        print(f"Input directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Target size: {args.target_size}")
        print(f"Target spacing: {args.target_spacing}")
        
        # Scan and process data
        processor.scan_data_directory()
        processor.preprocess_data(max_files=args.max_files)
        
        # Analyze data distribution
        processor.analyze_data_distribution()
        
        # Save processed data
        output_path = processor.save_processed_data(args.output_dir)
        
        # Create sample batches
        processor.create_sample_batches(args.output_dir, batch_size=args.batch_size)
        
        # Generate visualization if requested
        if args.visualize:
            processor.visualize_preprocessing(args.output_dir)
        
        print(f"\nProcessing completed successfully!")
        print(f"Processed data saved to: {output_path}")
        print("\nNext step: Run 'python train_brain_mapper.py' to train the model")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
