#!/usr/bin/env python3
"""
Create data splits for brain mapper training
This script organizes processed brain data into train/val/test splits
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import random

def create_data_splits(data_dir, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create train/validation/test splits from processed brain data
    
    Args:
        data_dir: Directory containing processed brain data
        test_size: Proportion of data for testing
        val_size: Proportion of remaining data for validation
        random_state: Random seed for reproducibility
    """
    
    data_dir = Path(data_dir)
    
    # Look for processed files
    processed_dir = data_dir / 'processed_files'
    if not processed_dir.exists():
        print(f"Error: Processed files directory not found: {processed_dir}")
        print("Please run the brain data preprocessing script first.")
        return False
    
    # Find all processed .npy files
    processed_files = list(processed_dir.glob("*.npy"))
    
    if len(processed_files) == 0:
        print(f"Error: No processed .npy files found in {processed_dir}")
        return False
    
    print(f"Found {len(processed_files)} processed files")
    
    # Create subject data entries
    subjects = []
    for i, file_path in enumerate(processed_files):
        # Extract subject info from filename
        filename = file_path.stem
        subject_id = f"subject_{i:03d}"
        
        # For demonstration, randomly assign labels (0=control, 1=case)
        # In real scenario, you'd have actual labels
        np.random.seed(random_state + i)  # Ensure reproducible labels
        label = np.random.choice([0, 1])
        
        subject_entry = {
            "image": str(file_path),
            "label": int(label),
            "subject_id": subject_id,
            "original_filename": filename
        }
        
        subjects.append(subject_entry)
    
    print(f"Created {len(subjects)} subject entries")
    
    # Balance classes for better training
    controls = [s for s in subjects if s['label'] == 0]
    cases = [s for s in subjects if s['label'] == 1]
    
    print(f"Class distribution:")
    print(f"  - Controls: {len(controls)}")
    print(f"  - Cases: {len(cases)}")
    
    # Create balanced dataset if needed
    min_class_size = min(len(controls), len(cases))
    if len(controls) != len(cases):
        print(f"Balancing classes to {min_class_size} samples each")
        
        random.seed(random_state)
        controls = random.sample(controls, min_class_size)
        cases = random.sample(cases, min_class_size)
    
    # Combine balanced dataset
    balanced_subjects = controls + cases
    
    print(f"Final balanced dataset: {len(balanced_subjects)} subjects")
    print(f"  - Controls: {len([s for s in balanced_subjects if s['label'] == 0])}")
    print(f"  - Cases: {len([s for s in balanced_subjects if s['label'] == 1])}")
    
    # Create train/test split
    train_val_subjects, test_subjects = train_test_split(
        balanced_subjects, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[s['label'] for s in balanced_subjects]
    )
    
    # Create train/validation split
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size / (1 - test_size),  # Adjust val_size for remaining data
        random_state=random_state,
        stratify=[s['label'] for s in train_val_subjects]
    )
    
    print(f"\nData splits created:")
    print(f"  - Train: {len(train_subjects)} subjects")
    print(f"  - Validation: {len(val_subjects)} subjects")
    print(f"  - Test: {len(test_subjects)} subjects")
    
    # Load processing metadata if available
    metadata_file = data_dir / 'processing_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            processing_metadata = json.load(f)
        target_size = processing_metadata.get('target_size', [128, 128, 128])
    else:
        target_size = [128, 128, 128]  # Default
        print("Warning: Processing metadata not found, using default target size")
    
    # Create data splits structure
    data_splits = {
        'train_data': train_subjects,
        'val_data': val_subjects,
        'test_data': test_subjects,
        'target_size': target_size,
        'total_subjects': len(balanced_subjects),
        'class_balance': {
            'controls': len([s for s in balanced_subjects if s['label'] == 0]),
            'cases': len([s for s in balanced_subjects if s['label'] == 1])
        },
        'split_ratios': {
            'train': len(train_subjects) / len(balanced_subjects),
            'val': len(val_subjects) / len(balanced_subjects),
            'test': len(test_subjects) / len(balanced_subjects)
        },
        'random_state': random_state
    }
    
    # Save data splits
    splits_file = data_dir / 'data_splits.json'
    with open(splits_file, 'w') as f:
        json.dump(data_splits, f, indent=2)
    
    print(f"\nData splits saved to: {splits_file}")
    
    # Create a simple ground truth file for demonstration
    create_demo_ground_truth(data_dir, target_size)
    
    return True

def create_demo_ground_truth(data_dir, target_size):
    """Create a demonstration ground truth file"""
    
    # Create simulated ground truth effects
    ground_truth = {
        'brain_shape': target_size,
        'effects_info': [
            {
                'center': [target_size[0]//2 + 10, target_size[1]//2, target_size[2]//2],
                'size': 8,
                'effect_strength': 0.3,
                'description': 'Simulated effect region 1'
            },
            {
                'center': [target_size[0]//2 - 10, target_size[1]//2 + 5, target_size[2]//2 - 5],
                'size': 6,
                'effect_strength': 0.25,
                'description': 'Simulated effect region 2'
            }
        ],
        'effect_type': 'demonstration',
        'note': 'This is simulated ground truth for demonstration purposes'
    }
    
    gt_file = data_dir / 'ground_truth_effects.json'
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Demo ground truth saved to: {gt_file}")

def verify_data_splits(data_dir):
    """Verify that data splits are valid"""
    
    data_dir = Path(data_dir)
    splits_file = data_dir / 'data_splits.json'
    
    if not splits_file.exists():
        print(f"Error: Data splits file not found: {splits_file}")
        return False
    
    with open(splits_file, 'r') as f:
        data_splits = json.load(f)
    
    print("Verifying data splits...")
    
    # Check that all referenced files exist
    all_subjects = (data_splits['train_data'] + 
                   data_splits['val_data'] + 
                   data_splits['test_data'])
    
    missing_files = []
    for subject in all_subjects:
        image_path = Path(subject['image'])
        if not image_path.exists():
            missing_files.append(str(image_path))
    
    if missing_files:
        print(f"Error: {len(missing_files)} referenced files are missing:")
        for f in missing_files[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    
    print("✓ All referenced files exist")
    
    # Check class balance
    for split_name in ['train_data', 'val_data', 'test_data']:
        split_data = data_splits[split_name]
        controls = len([s for s in split_data if s['label'] == 0])
        cases = len([s for s in split_data if s['label'] == 1])
        print(f"✓ {split_name}: {len(split_data)} subjects ({controls} controls, {cases} cases)")
    
    print("✓ Data splits verification completed successfully")
    return True

def main():
    parser = argparse.ArgumentParser(description='Create data splits for brain mapper training')
    parser.add_argument('data_dir', help='Directory containing processed brain data')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Proportion of remaining data for validation (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing data splits')
    
    args = parser.parse_args()
    
    print("=== Brain Data Splits Creator ===")
    print(f"Data directory: {args.data_dir}")
    
    # Validate input directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return 1
    
    if args.verify_only:
        success = verify_data_splits(args.data_dir)
    else:
        print(f"Test size: {args.test_size}")
        print(f"Validation size: {args.val_size}")
        print(f"Random state: {args.random_state}")
        
        success = create_data_splits(
            args.data_dir, 
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state
        )
    
    if success:
        print("\n✓ Data splits ready for training!")
        print("You can now run: python train_brain_mapper.py --data_dir ./processed_brain_data --interpret --evaluate")
    else:
        print("\n✗ Failed to create/verify data splits")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
