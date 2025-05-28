import os
import argparse
from pathlib import Path


def get_files_in_directory(directory):
    """Get all files in a directory and return their basenames."""
    files = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            files.append(file)
    return set(files)


def compare_directories(dir1, dir2):
    """Compare files in two directories and print differences."""
    # Get files from both directories
    files_in_dir1 = get_files_in_directory(dir1)
    files_in_dir2 = get_files_in_directory(dir2)
    
    # Find differences
    files_only_in_dir1 = files_in_dir1 - files_in_dir2
    files_only_in_dir2 = files_in_dir2 - files_in_dir1
    
    # Print results
    print(f"\nFiles only in {dir1}:")
    if files_only_in_dir1:
        for file in sorted(files_only_in_dir1):
            print(f"  - {file}")
    else:
        print("  None")
    
    print(f"\nFiles only in {dir2}:")
    if files_only_in_dir2:
        for file in sorted(files_only_in_dir2):
            print(f"  - {file}")
    else:
        print("  None")


def main():
    parser = argparse.ArgumentParser(description='Compare image files between two folders')
    parser.add_argument('dir1', help='First directory path')
    parser.add_argument('dir2', help='Second directory path')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.isdir(args.dir1):
        print(f"Error: {args.dir1} is not a valid directory")
        return
    
    if not os.path.isdir(args.dir2):
        print(f"Error: {args.dir2} is not a valid directory")
        return
    
    # Compare directories
    compare_directories(args.dir1, args.dir2)


if __name__ == "__main__":
    main()