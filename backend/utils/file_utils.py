"""File utilities for the deepfake detection pipeline."""
import os
from pathlib import Path


def ensure_dir(directory):
    """Ensure a directory exists."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory
