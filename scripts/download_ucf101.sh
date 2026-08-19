#!/bin/bash
# Download and extract UCF101 dataset and train/test split annotations.
#
# UCF101 dataset:
#   - 13,320 videos across 101 action classes
#   - ~6.5 GB compressed
#
# Usage:
#   bash scripts/download_ucf101.sh [data_dir]
#   Default data_dir: ./data

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "=== Downloading UCF101 Dataset ==="
echo "This may take a while (~6.5 GB)..."

# Download dataset
if [ ! -d "UCF-101" ]; then
    wget -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101.rar -O UCF101.rar
    echo "Extracting UCF101..."
    unrar x UCF101.rar
    rm UCF101.rar
    echo "UCF101 extracted to $DATA_DIR/UCF-101"
else
    echo "UCF-101 directory already exists, skipping download."
fi

# Download train/test split annotations
if [ ! -d "ucfTrainTestlist" ]; then
    echo ""
    echo "=== Downloading Train/Test Split Annotations ==="
    wget -q --show-progress https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip \
        -O UCF101TrainTestSplits.zip
    unzip -q UCF101TrainTestSplits.zip
    rm UCF101TrainTestSplits.zip
    echo "Annotations extracted to $DATA_DIR/ucfTrainTestlist"
else
    echo "ucfTrainTestlist directory already exists, skipping download."
fi

echo ""
echo "=== Done ==="
echo "Dataset structure:"
echo "  $DATA_DIR/UCF-101/          - Video files (101 class folders)"
echo "  $DATA_DIR/ucfTrainTestlist/  - Train/test split annotations"
echo ""
echo "Total videos:"
find UCF-101 -name "*.avi" | wc -l
