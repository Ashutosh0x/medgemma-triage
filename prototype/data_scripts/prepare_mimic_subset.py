#!/usr/bin/env python3
"""
MIMIC-CXR Subset Preparation Script
=====================================

This script prepares a subset of MIMIC-CXR for training and evaluation
of the CXR triage model.

Prerequisites:
- MIMIC-CXR-JPG dataset downloaded (see download_datasets.sh)
- Python 3.10+
- pandas, pydicom, Pillow

Usage:
    python prepare_mimic_subset.py --mimic_root /path/to/mimic-cxr-jpg \
                                   --output_dir ./data/processed \
                                   --subset_size 1000
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Urgent conditions (for triage classification)
URGENT_CONDITIONS = [
    'Pneumonia',
    'Lung Opacity',
    'Consolidation',
    'Pneumothorax',
    'Pleural Effusion',
    'Cardiomegaly',
    'Edema',
    'Atelectasis',
]

# Non-urgent conditions
NON_URGENT_CONDITIONS = [
    'No Finding',
    'Support Devices',
]


def load_mimic_metadata(mimic_root: Path) -> 'pd.DataFrame':
    """Load MIMIC-CXR metadata from CSV files."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas")
    
    # Load split file
    split_file = mimic_root / 'mimic-cxr-2.0.0-split.csv'
    chexpert_file = mimic_root / 'mimic-cxr-2.0.0-chexpert.csv'
    metadata_file = mimic_root / 'mimic-cxr-2.0.0-metadata.csv'
    
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not chexpert_file.exists():
        raise FileNotFoundError(f"CheXpert labels not found: {chexpert_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    logger.info("Loading MIMIC-CXR metadata...")
    
    splits = pd.read_csv(split_file)
    labels = pd.read_csv(chexpert_file)
    metadata = pd.read_csv(metadata_file)
    
    # Merge dataframes
    df = splits.merge(labels, on=['subject_id', 'study_id'], how='left')
    df = df.merge(metadata, on=['dicom_id', 'subject_id', 'study_id'], how='left')
    
    logger.info(f"Loaded {len(df)} records")
    return df


def classify_urgency(row: 'pd.Series') -> str:
    """Classify a study as urgent or non-urgent based on CheXpert labels."""
    for condition in URGENT_CONDITIONS:
        if condition in row and row[condition] == 1.0:
            return 'urgent'
    return 'non-urgent'


def get_primary_finding(row: 'pd.Series') -> str:
    """Get the primary finding for a study."""
    for condition in URGENT_CONDITIONS:
        if condition in row and row[condition] == 1.0:
            return condition
    for condition in NON_URGENT_CONDITIONS:
        if condition in row and row[condition] == 1.0:
            return condition
    return 'Unknown'


def create_subset(
    df: 'pd.DataFrame',
    mimic_root: Path,
    output_dir: Path,
    subset_size: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """Create a balanced subset for training."""
    random.seed(seed)
    
    logger.info("Creating balanced subset...")
    
    # Add urgency labels
    df['urgency'] = df.apply(classify_urgency, axis=1)
    df['primary_finding'] = df.apply(get_primary_finding, axis=1)
    
    # Filter to frontal views only (PA and AP)
    frontal_views = df[df['ViewPosition'].isin(['PA', 'AP'])]
    logger.info(f"Frontal views: {len(frontal_views)}")
    
    # Balance urgent vs non-urgent
    urgent = frontal_views[frontal_views['urgency'] == 'urgent']
    non_urgent = frontal_views[frontal_views['urgency'] == 'non-urgent']
    
    sample_size = min(subset_size // 2, len(urgent), len(non_urgent))
    
    urgent_sample = urgent.sample(n=sample_size, random_state=seed)
    non_urgent_sample = non_urgent.sample(n=sample_size, random_state=seed)
    
    subset_df = pd.concat([urgent_sample, non_urgent_sample])
    subset_df = subset_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    logger.info(f"Subset size: {len(subset_df)} (urgent: {sample_size}, non-urgent: {sample_size})")
    
    # Split into train/val/test
    n_train = int(len(subset_df) * train_ratio)
    n_val = int(len(subset_df) * val_ratio)
    
    train_df = subset_df[:n_train]
    val_df = subset_df[n_train:n_train + n_val]
    test_df = subset_df[n_train + n_val:]
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to records
    def df_to_records(split_df: 'pd.DataFrame', split_name: str) -> List[Dict]:
        records = []
        for _, row in split_df.iterrows():
            # Construct image path
            subject_id = f"p{str(row['subject_id'])[:2]}/p{row['subject_id']}"
            study_id = f"s{row['study_id']}"
            dicom_id = f"{row['dicom_id']}.jpg"
            img_path = mimic_root / 'files' / subject_id / study_id / dicom_id
            
            record = {
                'id': f"{row['subject_id']}_{row['study_id']}_{row['dicom_id']}",
                'image_path': str(img_path),
                'subject_id': int(row['subject_id']),
                'study_id': int(row['study_id']),
                'dicom_id': row['dicom_id'],
                'split': split_name,
                'urgency': row['urgency'],
                'primary_finding': row['primary_finding'],
                'view_position': row['ViewPosition'],
            }
            records.append(record)
        return records
    
    splits = {
        'train': df_to_records(train_df, 'train'),
        'val': df_to_records(val_df, 'val'),
        'test': df_to_records(test_df, 'test'),
    }
    
    # Save to JSON files
    for split_name, records in splits.items():
        output_file = output_dir / f'{split_name}.jsonl'
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Saved {len(records)} records to {output_file}")
    
    # Save combined metadata
    all_records = splits['train'] + splits['val'] + splits['test']
    metadata = {
        'total_samples': len(all_records),
        'train_samples': len(splits['train']),
        'val_samples': len(splits['val']),
        'test_samples': len(splits['test']),
        'urgent_conditions': URGENT_CONDITIONS,
        'non_urgent_conditions': NON_URGENT_CONDITIONS,
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return splits


def create_sample_dataset(output_dir: Path) -> None:
    """Create a sample dataset using public CC0 images for development."""
    logger.info("Creating sample dataset with CC0 images...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data for development (using placeholder data)
    sample_records = [
        {
            'id': 'sample_001',
            'image_url': 'https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png',
            'urgency': 'non-urgent',
            'primary_finding': 'No Finding',
            'split': 'test',
            'note': 'CC0 licensed sample image from Wikimedia Commons'
        },
    ]
    
    # Save sample test set
    with open(output_dir / 'sample_test.jsonl', 'w') as f:
        for record in sample_records:
            f.write(json.dumps(record) + '\n')
    
    logger.info(f"Saved {len(sample_records)} sample records to {output_dir / 'sample_test.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description='Prepare MIMIC-CXR subset for training')
    parser.add_argument('--mimic_root', type=str, help='Path to MIMIC-CXR-JPG root directory')
    parser.add_argument('--output_dir', type=str, default='./data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--subset_size', type=int, default=1000,
                        help='Number of samples in subset')
    parser.add_argument('--sample_only', action='store_true',
                        help='Create sample dataset only (no MIMIC data required)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if args.sample_only:
        create_sample_dataset(output_dir)
        return
    
    if not args.mimic_root:
        logger.error("--mimic_root is required unless using --sample_only")
        parser.print_help()
        return
    
    mimic_root = Path(args.mimic_root)
    if not mimic_root.exists():
        raise FileNotFoundError(f"MIMIC-CXR root not found: {mimic_root}")
    
    df = load_mimic_metadata(mimic_root)
    create_subset(df, mimic_root, output_dir, args.subset_size, seed=args.seed)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
