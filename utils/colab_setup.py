"""Google Colab environment setup — Drive mount and path constants."""

from pathlib import Path

DRIVE_PATH = Path('/content/drive/MyDrive/Dissertation_Colab')
DATA_DIR = DRIVE_PATH / 'data'
CHECKPOINT_DIR = DRIVE_PATH / 'checkpoints'
OUTPUT_DIR = DRIVE_PATH / 'outputs'


def setup_drive():
    """Mount Google Drive and create working directories."""
    from google.colab import drive
    drive.mount('/content/drive')

    for dir_path in [DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Data directory:       {DATA_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"Output directory:     {OUTPUT_DIR}")
