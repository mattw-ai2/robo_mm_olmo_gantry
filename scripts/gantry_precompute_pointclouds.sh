#!/bin/bash
#SBATCH --job-name=precompute_pointclouds
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pointcloud_precompute_%j.out
#SBATCH --error=logs/pointcloud_precompute_%j.err

# =============================================================================
# Gantry Script: Precompute Point Clouds from VIDA Dataset
# =============================================================================
#
# This script runs VGGT depth estimation and point cloud accumulation on the
# VIDA robot navigation dataset. For each keyframe in a trajectory, it creates
# an ACCUMULATED point cloud from all frames up to that keyframe.
#
# Accumulation scheme (for trajectory with frames 1-10):
#   - Keyframe 1: Point cloud from frame 1 only
#   - Keyframe 2: Accumulated point cloud from frames 1 + 2
#   - Keyframe 3: Accumulated point cloud from frames 1 + 2 + 3
#   - ...
#   - Keyframe N: Accumulated point cloud from frames 1 + 2 + ... + N
#
# Usage:
#   sbatch scripts/gantry_precompute_pointclouds.sh [TASK_TYPE] [SPLIT]
#
# Examples:
#   sbatch scripts/gantry_precompute_pointclouds.sh ObjectNavType train
#   sbatch scripts/gantry_precompute_pointclouds.sh HardObjectNavType train
#   sbatch scripts/gantry_precompute_pointclouds.sh SimpleExploreHouse validation
#
# Or run directly:
#   bash scripts/gantry_precompute_pointclouds.sh ObjectNavType train
#
# =============================================================================

set -e

# Parse arguments
TASK_TYPE="${1:-ObjectNavType}"
SPLIT="${2:-train}"

# Configuration
BASE_INPUT_DIR="/weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor"
BASE_OUTPUT_DIR="/weka/prior/mattw/data/robot_point_clouds"
MAX_POINTS_PER_FRAME=5000
DEVICE="cuda"

# Construct paths
INPUT_DIR="${BASE_INPUT_DIR}/${TASK_TYPE}/${SPLIT}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${TASK_TYPE}/${SPLIT}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=============================================="
echo "  Point Cloud Preprocessing"
echo "=============================================="
echo "Task Type:         ${TASK_TYPE}"
echo "Split:             ${SPLIT}"
echo "Input Dir:         ${INPUT_DIR}"
echo "Output Dir:        ${OUTPUT_DIR}"
echo "Max Points/Frame:  ${MAX_POINTS_PER_FRAME}"
echo "Device:            ${DEVICE}"
echo "=============================================="

# Check input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Count houses
NUM_HOUSES=$(ls -d "${INPUT_DIR}"/*/ 2>/dev/null | wc -l)
echo "Found ${NUM_HOUSES} houses to process"

# Activate conda environment with VGGT
cd /weka/prior/mattw/robo_mm_olmo
source ~/.bashrc  # Ensure conda is available
conda activate vggt

# Run preprocessing
echo ""
echo "Starting preprocessing at $(date)"
echo ""

python scripts/precompute_vggt_depth.py \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_points_per_frame ${MAX_POINTS_PER_FRAME} \
    --device ${DEVICE} \
    --resume

echo ""
echo "Preprocessing complete at $(date)"
echo ""

# Print output stats
if [ -d "${OUTPUT_DIR}" ]; then
    NUM_OUTPUT_FILES=$(find "${OUTPUT_DIR}" -name "point_clouds.hdf5" | wc -l)
    TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" | cut -f1)
    echo "=============================================="
    echo "  Output Statistics"
    echo "=============================================="
    echo "Output files: ${NUM_OUTPUT_FILES}"
    echo "Total size:   ${TOTAL_SIZE}"
    echo "=============================================="
fi

