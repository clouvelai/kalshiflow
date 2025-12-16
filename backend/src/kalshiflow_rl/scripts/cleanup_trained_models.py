#!/usr/bin/env python3
"""
Cleanup script for trained models directory.
Removes all models except the one specified in CURRENT_MODEL.json.
"""

import json
import os
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def load_current_model_config():
    """Load the CURRENT_MODEL.json configuration."""
    config_path = Path(__file__).parent.parent / "CURRENT_MODEL.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"CURRENT_MODEL.json not found at {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_models_to_keep(config):
    """Extract all model paths that should be kept from the config."""
    models_to_keep = set()
    
    # Add current model - be very specific about the exact file/directory name
    if 'current_model' in config:
        if 'full_path' in config['current_model']:
            # Handle both absolute and relative paths
            model_path = config['current_model']['full_path']
            # Extract just the filename/dirname
            if '/' in model_path:
                models_to_keep.add(Path(model_path).name)
            else:
                models_to_keep.add(model_path)
        
        # Also check model_file field 
        if 'model_file' in config['current_model']:
            models_to_keep.add(config['current_model']['model_file'])
    
    # Add specific models from version history that we want to keep
    # Only keep the exact model mentioned, not similar ones
    if 'version_history' in config:
        for version in config['version_history']:
            if 'model_path' in version:
                model_path = version['model_path']
                if 'trained_models/' in model_path:
                    # Get exact directory name without trained_models/ prefix
                    dir_name = model_path.split('trained_models/')[-1].rstrip('/')
                    # Only add if it's the exact historical model we care about
                    if 'session9_ppo_20251211_221054' in dir_name:
                        models_to_keep.add('session9_ppo_20251211_221054')
    
    return models_to_keep


def cleanup_trained_models(dry_run=True, verbose=True):
    """
    Clean up trained models directory, keeping only the current model.
    
    Args:
        dry_run: If True, only show what would be deleted without actually deleting
        verbose: If True, print detailed output
    
    Returns:
        Dict with cleanup statistics
    """
    # Load configuration
    config = load_current_model_config()
    models_to_keep = get_models_to_keep(config)
    
    # Find trained_models directory
    trained_models_dir = Path(__file__).parent.parent.parent.parent / "trained_models"
    
    if not trained_models_dir.exists():
        print(f"Trained models directory not found: {trained_models_dir}")
        return {"error": "Directory not found"}
    
    # Scan directory
    all_items = list(trained_models_dir.iterdir())
    
    stats = {
        "total_items": len(all_items),
        "directories_to_delete": [],
        "files_to_delete": [],
        "items_to_keep": [],
        "total_size_to_delete": 0,
        "total_size_to_keep": 0
    }
    
    # Categorize items
    for item in all_items:
        item_name = item.name
        
        # Skip hidden files and special directories
        if item_name.startswith('.') or item_name in ['__pycache__', '.DS_Store']:
            continue
        
        # Check if this item should be kept (exact match only)
        should_keep = item_name in models_to_keep
        
        # Calculate size
        if item.is_dir():
            size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
        else:
            size = item.stat().st_size if item.is_file() else 0
        
        if should_keep:
            stats["items_to_keep"].append({
                "path": str(item),
                "name": item_name,
                "type": "directory" if item.is_dir() else "file",
                "size_mb": size / (1024 * 1024)
            })
            stats["total_size_to_keep"] += size
        else:
            if item.is_dir():
                stats["directories_to_delete"].append({
                    "path": str(item),
                    "name": item_name,
                    "size_mb": size / (1024 * 1024)
                })
            else:
                stats["files_to_delete"].append({
                    "path": str(item),
                    "name": item_name,
                    "size_mb": size / (1024 * 1024)
                })
            stats["total_size_to_delete"] += size
    
    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("TRAINED MODELS CLEANUP SUMMARY")
        print("="*70)
        print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL DELETION'}")
        print(f"Current model: {models_to_keep}")
        print(f"\nTotal items found: {stats['total_items']}")
        print(f"Items to keep: {len(stats['items_to_keep'])}")
        print(f"Directories to delete: {len(stats['directories_to_delete'])}")
        print(f"Files to delete: {len(stats['files_to_delete'])}")
        print(f"Total size to delete: {stats['total_size_to_delete'] / (1024 * 1024):.2f} MB")
        print(f"Total size to keep: {stats['total_size_to_keep'] / (1024 * 1024):.2f} MB")
        
        if stats["items_to_keep"]:
            print("\n" + "-"*70)
            print("ITEMS TO KEEP:")
            for item in stats["items_to_keep"]:
                print(f"  ✓ {item['name']} ({item['type']}, {item['size_mb']:.2f} MB)")
        
        if stats["directories_to_delete"]:
            print("\n" + "-"*70)
            print("DIRECTORIES TO DELETE:")
            for item in sorted(stats["directories_to_delete"], key=lambda x: x['size_mb'], reverse=True):
                print(f"  ✗ {item['name']} ({item['size_mb']:.2f} MB)")
        
        if stats["files_to_delete"]:
            print("\n" + "-"*70)
            print("FILES TO DELETE:")
            for item in sorted(stats["files_to_delete"], key=lambda x: x['size_mb'], reverse=True):
                print(f"  ✗ {item['name']} ({item['size_mb']:.2f} MB)")
    
    # Perform actual deletion if not dry run
    if not dry_run:
        deleted_count = 0
        failed_deletions = []
        
        # Delete directories
        for item in stats["directories_to_delete"]:
            try:
                shutil.rmtree(item["path"])
                deleted_count += 1
                if verbose:
                    print(f"Deleted directory: {item['name']}")
            except Exception as e:
                failed_deletions.append({"path": item["path"], "error": str(e)})
                if verbose:
                    print(f"Failed to delete directory {item['name']}: {e}")
        
        # Delete files
        for item in stats["files_to_delete"]:
            try:
                os.remove(item["path"])
                deleted_count += 1
                if verbose:
                    print(f"Deleted file: {item['name']}")
            except Exception as e:
                failed_deletions.append({"path": item["path"], "error": str(e)})
                if verbose:
                    print(f"Failed to delete file {item['name']}: {e}")
        
        stats["deleted_count"] = deleted_count
        stats["failed_deletions"] = failed_deletions
        
        if verbose:
            print("\n" + "="*70)
            print(f"DELETION COMPLETE: {deleted_count} items deleted")
            if failed_deletions:
                print(f"Failed to delete {len(failed_deletions)} items")
            print(f"Space freed: {stats['total_size_to_delete'] / (1024 * 1024):.2f} MB")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Clean up trained models directory, keeping only the current model"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (default is dry run)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    # Run cleanup
    stats = cleanup_trained_models(
        dry_run=not args.execute,
        verbose=not args.quiet
    )
    
    # Save log
    if not args.execute:
        print("\n" + "="*70)
        print("This was a DRY RUN. No files were deleted.")
        print("To actually delete files, run with --execute flag:")
        print("  python cleanup_trained_models.py --execute")
    else:
        # Save deletion log
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"model_cleanup_{timestamp}.json"
        
        with open(log_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"\nDeletion log saved to: {log_file}")


if __name__ == "__main__":
    main()