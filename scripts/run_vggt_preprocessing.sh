#!/bin/bash
# VGGT Point Cloud Preprocessing Script
# 
# This script runs the VGGT preprocessing to generate accumulated point clouds
# from the VIDA robot dataset. Run this BEFORE training with point clouds.
#
# Usage:
#   ./scripts/run_vggt_preprocessing.sh [task_type] [split] [num_gpus]
#
# Examples:
#   # Single GPU:
#   ./scripts/run_vggt_preprocessing.sh ObjectNavType train
#
#   # Multi-GPU (4 GPUs):
#   ./scripts/run_vggt_preprocessing.sh ObjectNavType train 4
#
#   # With environment variables:
#   NUM_GPUS=8 RESUME=1 ./scripts/run_vggt_preprocessing.sh ObjectNavType train
#
# Environment variables:
#   NUM_GPUS      - Number of GPUs to use (default: 1)
#   RESUME        - Set to 1 to skip already processed houses
#   MAX_POINTS    - Max points per cloud (default: 50000)
#   MAX_KEYFRAMES - Max keyframes per episode (default: 10)
#   NUM_HOUSES    - Limit number of houses (for testing)
#   START_HOUSE   - Start index for house range
#   END_HOUSE     - End index for house range
#
# The script will process the dataset and save point clouds to:
#   $OUTPUT_BASE_DIR/{task_type}/{split}/

set -e

# Configuration
TASK_TYPE="${1:-ObjectNavType}"
SPLIT="${2:-train}"
NUM_GPUS="${3:-${NUM_GPUS:-1}}"

INPUT_BASE_DIR="/weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor"
OUTPUT_BASE_DIR="${ROBOT_POINT_CLOUD_DIR:-/weka/prior/mattw/data/robot_point_clouds}"

INPUT_DIR="${INPUT_BASE_DIR}/${TASK_TYPE}/${SPLIT}"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${TASK_TYPE}/${SPLIT}"

# Parameters
MAX_POINTS="${MAX_POINTS:-50000}"
MAX_KEYFRAMES="${MAX_KEYFRAMES:-10}"
RESUME="${RESUME:-0}"

echo "=============================================="
echo "VGGT Point Cloud Preprocessing"
echo "=============================================="
echo "Task Type:        ${TASK_TYPE}"
echo "Split:            ${SPLIT}"
echo "Input Directory:  ${INPUT_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Max Points:       ${MAX_POINTS}"
echo "Max Keyframes:    ${MAX_KEYFRAMES}"
echo "Num GPUs:         ${NUM_GPUS}"
if [ "${RESUME}" = "1" ]; then
    echo "Resume Mode:      ENABLED"
fi
if [ -n "${NUM_HOUSES}" ]; then
    echo "Num Houses:       ${NUM_HOUSES} (limit)"
fi
if [ -n "${START_HOUSE}" ] || [ -n "${END_HOUSE}" ]; then
    echo "House Range:      ${START_HOUSE:-0} to ${END_HOUSE:-end}"
fi
echo "=============================================="

# Check input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory does not exist: ${INPUT_DIR}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command
CMD="python scripts/precompute_vggt_depth.py \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --max_points ${MAX_POINTS} \
    --max_keyframes ${MAX_KEYFRAMES} \
    --num_gpus ${NUM_GPUS}"

if [ "${RESUME}" = "1" ]; then
    CMD="${CMD} --resume"
fi

if [ -n "${NUM_HOUSES}" ]; then
    CMD="${CMD} --num_houses ${NUM_HOUSES}"
fi

if [ -n "${START_HOUSE}" ]; then
    CMD="${CMD} --start_house ${START_HOUSE}"
fi

if [ -n "${END_HOUSE}" ]; then
    CMD="${CMD} --end_house ${END_HOUSE}"
fi

echo ""
echo "Running command:"
echo "${CMD}"
echo ""

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run preprocessing
eval ${CMD}

echo ""
echo "=============================================="
echo "Preprocessing complete!"
echo "Point clouds saved to: ${OUTPUT_DIR}"
echo ""
echo "To train with point clouds, use:"
echo "  --use_point_cloud --point_cloud_dir ${OUTPUT_BASE_DIR}"
echo "=============================================="
