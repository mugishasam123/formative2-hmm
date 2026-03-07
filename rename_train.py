#!/usr/bin/env python3
"""
Flatten the data/raw/train directory:
  - Move every file from each subfolder up to train/
  - Rename each file to: set1_<SubfolderName>_<original_filename>
  - Remove the now-empty subfolders
"""

import os
import shutil

TRAIN_DIR = os.path.join(os.path.dirname(__file__), "data", "raw", "train")


def rename_and_flatten(train_dir: str, dry_run: bool = False) -> None:
    subfolders = [
        entry for entry in os.scandir(train_dir)
        if entry.is_dir()
    ]

    if not subfolders:
        print("No subfolders found – nothing to do.")
        return

    for subfolder in subfolders:
        activity = subfolder.name.capitalize()  # e.g. jumping -> Jumping

        files = [
            f for f in os.scandir(subfolder.path)
            if f.is_file() and not f.name.startswith(".")
        ]

        for f in files:
            new_name = f"set1_{activity}_{f.name}"
            dest = os.path.join(train_dir, new_name)

            if dry_run:
                print(f"[DRY RUN] {f.path}  ->  {dest}")
            else:
                shutil.move(f.path, dest)
                print(f"Moved: {f.name}  ->  {new_name}")

        # Remove the subfolder (including hidden files like .DS_Store)
        if not dry_run:
            shutil.rmtree(subfolder.path)
            print(f"Removed subfolder: {subfolder.name}/")

    print("\nDone.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flatten and rename train data files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without moving/renaming any files.",
    )
    args = parser.parse_args()

    rename_and_flatten(TRAIN_DIR, dry_run=args.dry_run)
