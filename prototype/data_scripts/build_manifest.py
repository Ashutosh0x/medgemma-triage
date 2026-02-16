#!/usr/bin/env python3
"""
Data Manifest Builder & Validator
=================================

Creates immutable provenance records for all dataset files.
Rejects files without proper source attribution.

Usage:
    python build_manifest.py --root ../data --output dataset_manifest.json
    python build_manifest.py --validate dataset_manifest.json
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


# Suspicious patterns that indicate synthetic/mock data
SYNTHETIC_PATTERNS = [
    r'synthetic', r'fake', r'mock', r'simulated', r'sim_',
    r'generated', r'artificial', r'aug_', r'demo_', r'test_sample'
]

# Required metadata for each file
REQUIRED_FIELDS = ['path', 'sha256', 'source', 'dataset', 'license']


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_suspicious(filepath: Path) -> tuple:
    """Check if file name/path contains suspicious synthetic patterns."""
    path_str = str(filepath).lower()
    for pattern in SYNTHETIC_PATTERNS:
        if re.search(pattern, path_str):
            return True, pattern
    return False, None


def build_manifest(
    root: Path,
    source: str = "unknown",
    dataset: str = "unknown",
    license_url: str = "unknown",
    output: Path = Path("dataset_manifest.json"),
    extensions: List[str] = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']
) -> Dict:
    """
    Build manifest for all image files in directory.
    
    Args:
        root: Root directory to scan
        source: Dataset source URL
        dataset: Dataset name/accession
        license_url: License/TOU URL
        output: Output manifest file path
        extensions: File extensions to include
    
    Returns:
        Manifest dict
    """
    manifest = {
        "created": datetime.now().isoformat(),
        "root": str(root),
        "source": source,
        "dataset": dataset,
        "license": license_url,
        "total_files": 0,
        "total_bytes": 0,
        "files": [],
        "warnings": [],
    }
    
    print(f"Scanning {root}...")
    
    for ext in extensions:
        for filepath in root.rglob(f'*{ext}'):
            # Check for synthetic patterns
            suspicious, pattern = is_suspicious(filepath)
            if suspicious:
                warning = f"SUSPICIOUS: {filepath} matches pattern '{pattern}'"
                manifest["warnings"].append(warning)
                print(f"[WARN] {warning}")
                continue  # Skip suspicious files
            
            # Build file entry
            entry = {
                "path": str(filepath.relative_to(root)),
                "sha256": sha256_file(filepath),
                "size": filepath.stat().st_size,
                "source": source,
                "dataset": dataset,
                "license": license_url,
            }
            
            manifest["files"].append(entry)
            manifest["total_bytes"] += entry["size"]
    
    manifest["total_files"] = len(manifest["files"])
    
    # Save manifest
    with open(output, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"[OK] Manifest saved to {output}")
    print(f"  Total files: {manifest['total_files']}")
    print(f"  Total size: {manifest['total_bytes'] / 1e6:.2f} MB")
    print(f"  Warnings: {len(manifest['warnings'])}")
    
    return manifest


def validate_manifest(manifest_path: Path) -> bool:
    """
    Validate a manifest file for completeness and integrity.
    
    Returns:
        True if valid, False otherwise
    """
    print(f"Validating {manifest_path}...")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    errors = []
    warnings = []
    
    # Check required top-level fields
    for field in ['source', 'dataset', 'license', 'files']:
        if field not in manifest or manifest[field] in ['unknown', '', None]:
            errors.append(f"Missing or unknown top-level field: {field}")
    
    # Check each file entry
    for entry in manifest.get('files', []):
        for field in REQUIRED_FIELDS:
            if field not in entry or entry[field] in ['unknown', '', None]:
                errors.append(f"File {entry.get('path', '?')} missing field: {field}")
        
        # Check for suspicious paths in entries
        suspicious, pattern = is_suspicious(Path(entry.get('path', '')))
        if suspicious:
            errors.append(f"File {entry['path']} contains synthetic pattern: {pattern}")
    
    # Check for warnings in manifest
    if manifest.get('warnings'):
        for w in manifest['warnings']:
            warnings.append(w)
    
    # Report results
    if errors:
        print("[FAIL] VALIDATION FAILED")
        for e in errors[:10]:  # Limit output
            print(f"  ERROR: {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        return False
    
    if warnings:
        print("[WARN] VALIDATION PASSED WITH WARNINGS")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("[OK] VALIDATION PASSED")
        print(f"  Files: {manifest['total_files']}")
        print(f"  Source: {manifest['source']}")
        print(f"  Dataset: {manifest['dataset']}")
    
    return True


def verify_file_integrity(manifest_path: Path, data_root: Path) -> bool:
    """
    Verify that files match their recorded SHA256 hashes.
    """
    print(f"Verifying file integrity...")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    errors = []
    verified = 0
    
    for entry in manifest.get('files', []):
        filepath = data_root / entry['path']
        
        if not filepath.exists():
            errors.append(f"Missing file: {entry['path']}")
            continue
        
        actual_hash = sha256_file(filepath)
        if actual_hash != entry['sha256']:
            errors.append(f"Hash mismatch: {entry['path']}")
            continue
        
        verified += 1
    
    if errors:
        print(f"[FAIL] INTEGRITY CHECK FAILED: {len(errors)} errors")
        for e in errors[:5]:
            print(f"  {e}")
        return False
    
    print(f"[OK] INTEGRITY CHECK PASSED: {verified} files verified")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or validate data manifest")
    parser.add_argument('--root', type=Path, help='Root directory to scan')
    parser.add_argument('--output', type=Path, default=Path('dataset_manifest.json'))
    parser.add_argument('--validate', type=Path, help='Validate existing manifest')
    parser.add_argument('--verify', type=Path, help='Verify file integrity against manifest')
    parser.add_argument('--source', default='unknown', help='Dataset source URL')
    parser.add_argument('--dataset', default='unknown', help='Dataset name')
    parser.add_argument('--license', default='unknown', help='License URL')
    
    args = parser.parse_args()
    
    if args.validate:
        success = validate_manifest(args.validate)
        sys.exit(0 if success else 1)
    
    if args.verify:
        if not args.root:
            print("--root required for integrity verification")
            sys.exit(1)
        success = verify_file_integrity(args.verify, args.root)
        sys.exit(0 if success else 1)
    
    if args.root:
        build_manifest(
            args.root,
            source=args.source,
            dataset=args.dataset,
            license_url=args.license,
            output=args.output,
        )
    else:
        parser.print_help()
