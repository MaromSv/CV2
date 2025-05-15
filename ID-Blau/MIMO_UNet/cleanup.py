import os
import shutil
import argparse

def cleanup_project(dry_run=True):
    """
    Clean up unnecessary files from the MIMO_UNet project
    
    Args:
        dry_run: If True, only print what would be removed without actually removing
    """
    # Files to keep (essential files)
    essential_files = [
        # Core model files
        'MIMOUNet.py',
        # Configuration
        'config/config.yaml',
        # Utilities
        'blur_utils.py',
        # Testing
        'test_blur_field.py',
        # Cleanup script
        'cleanup.py',
        # README and license
        'README.md',
        'LICENSE',
        # Git files
        '.git',
        '.gitignore'
    ]
    
    # Directories to keep
    essential_dirs = [
        'config',
        'utils',
        'experiments',  # For model weights
        'results',      # For output
        'dataset'       # For test data
    ]
    
    # Get all files and directories in the project
    all_items = []
    for root, dirs, files in os.walk('.'):
        if '.git' in root:
            continue
        for file in files:
            all_items.append(os.path.join(root, file))
        for dir in dirs:
            if dir != '.git':
                all_items.append(os.path.join(root, dir))
    
    # Determine what to remove
    to_remove = []
    for item in all_items:
        item = item[2:]  # Remove './' prefix
        keep = False
        
        # Check if item is an essential file
        for essential in essential_files:
            if item == essential or item.startswith(essential + os.sep):
                keep = True
                break
        
        # Check if item is in an essential directory
        if not keep:
            for essential in essential_dirs:
                if item == essential or item.startswith(essential + os.sep):
                    keep = True
                    break
        
        if not keep:
            to_remove.append(item)
    
    # Print or remove items
    if to_remove:
        print(f"Found {len(to_remove)} items to remove:")
        for item in to_remove:
            if dry_run:
                print(f"Would remove: {item}")
            else:
                print(f"Removing: {item}")
                if os.path.isfile(item):
                    os.remove(item)
                elif os.path.isdir(item):
                    shutil.rmtree(item)
    else:
        print("No unnecessary files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up unnecessary files from the MIMO_UNet project")
    parser.add_argument("--execute", action="store_true", help="Actually remove files (default is dry run)")
    args = parser.parse_args()
    
    cleanup_project(dry_run=not args.execute)