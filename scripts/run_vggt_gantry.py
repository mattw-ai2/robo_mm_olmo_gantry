#!/usr/bin/env python
"""
Gantry launcher for VGGT depth preprocessing.

Each job uses 1 GPU by default, so you can launch many parallel jobs to process
different house ranges efficiently.

Usage:
    # Process houses 0-1000:
    python scripts/run_vggt_gantry.py \
        "python scripts/precompute_vggt_depth.py --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train --output_dir /weka/prior/mattw/data/robot_point_clouds/ObjectNavType/train --start_house 0 --end_house 1000 --resume" \
        --name vggt-train-0

    # Launch multiple jobs in parallel:
    for i in $(seq 0 14); do
        start=$((i * 1000))
        end=$((start + 1000))
        python scripts/run_vggt_gantry.py \
            "python scripts/precompute_vggt_depth.py --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train --output_dir /weka/prior/mattw/data/robot_point_clouds/ObjectNavType/train --start_house $start --end_house $end --resume" \
            --name vggt-train-$i --preemptible
    done

    # With preemptible (cheaper, safe with --resume):
    python scripts/run_vggt_gantry.py \
        "python scripts/precompute_vggt_depth.py ..." \
        --name vggt-precompute --preemptible
"""

import argparse
import os
import subprocess
import sys

import click
from gantry.commands import run as gantry_run


# ============================================================================
# CONFIGURATION - Update these for your setup
# ============================================================================
# Path to your main code repo
SOURCE_REPO = "/weka/prior/mattw/robo_mm_olmo"
# Path to your gantry repo (separate git repo for beaker to clone)
GANTRY_REPO = "/weka/prior/mattw/robo_mm_olmo_gantry"
# Your GitHub token secret name in Beaker
GITHUB_TOKEN_SECRET = "MATTW_GITHUB_TOKEN"
# ============================================================================


def sync_code_to_gantry_repo(job_name: str):
    """Sync code to gantry repo and push to GitHub."""
    print("Syncing code to gantry repo...")
    
    # Ensure gantry repo exists
    if not os.path.exists(GANTRY_REPO):
        print(f"ERROR: Gantry repo not found at {GANTRY_REPO}")
        print("Please create it first - see script docstring for instructions")
        sys.exit(1)
    
    # Rsync code (excluding git, venv, cache, etc.)
    rsync_cmd = (
        f"rsync -rzv --delete "
        f"--exclude .git --exclude .gitmodules --exclude .idea "
        f"--exclude '*.html' --exclude '*.pyc' --exclude __pycache__ "
        f"--exclude .cache --exclude .pytest_cache --exclude scratch "
        f"--exclude '*.ipynb' --exclude .venv --exclude uv.lock "
        f"{SOURCE_REPO}/ {GANTRY_REPO}/"
    )
    subprocess.call(rsync_cmd, shell=True)
    
    # Commit and push if there are changes
    os.chdir(GANTRY_REPO)
    status = subprocess.check_output(["git", "status", "-s"]).decode("utf-8").strip()
    
    if status:
        print("Files changed, committing and pushing...")
        subprocess.call(["git", "add", "."])
        subprocess.call(["git", "commit", "-m", f"update for {job_name}"])
        subprocess.call(["git", "push"])
    else:
        print("No changes to commit")


def main():
    parser = argparse.ArgumentParser(
        description="Launch VGGT preprocessing on Beaker via Gantry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", help="Python command to run (e.g., 'python scripts/precompute_vggt_depth.py ...')")
    parser.add_argument("--name", required=True, help="Beaker job name")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--preemptible", action="store_true", help="Use preemptible instances")
    parser.add_argument("--priority", default="high", choices=["low", "normal", "high", "urgent"])
    parser.add_argument("--cluster", default=None, help="Specific cluster (default: saturn, neptune, rhea, triton)")
    parser.add_argument("--no_sync", action="store_true", help="Skip git sync (use existing gantry repo state)")
    args = parser.parse_args()

    command = args.command
    beaker_name = args.name

    # Sync code to gantry repo and change to it
    if not args.no_sync:
        sync_code_to_gantry_repo(beaker_name)
    
    # Gantry must run from the gantry repo directory
    os.chdir(GANTRY_REPO)

    # Gantry configuration
    gantry_kwargs = dict(
        name=beaker_name,
        task_name=beaker_name,
        priority=args.priority,
        budget="ai2/oe-mm",
        gpus=args.gpus,
        shared_memory="16GiB",
        beaker_image="chrisc/molmo-torch2.6.0-cuda12.6-video",
        workspace="ai2/robo-molmo",
        gh_token_secret=GITHUB_TOKEN_SECRET,
        description=f"VGGT preprocessing: {command[:100]}...",
        preemptible=args.preemptible,
    )

    env = dict(
        OMP_NUM_THREADS="8",
        # Weka access
        WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000",
        WEKA_PROFILE="weka",
    )

    env_secret = dict(
        HF_ACCESS_TOKEN="MATTW_HF_TOKEN",
        BEAKER_TOKEN="MATTW_BEAKER_TOKEN",
    )

    # Clusters
    if args.cluster:
        clusters = [args.cluster]
    else:
        clusters = ["ai2/saturn", "ai2/neptune", "ai2/rhea", "ai2/triton"]

    # Weka mounts
    gantry_kwargs["weka"] = [
        "oe-training-default:/weka/oe-training-default",
        "prior-default:/weka/prior",
    ]

    # Build the full command with setup
    # Note: Gantry clones the repo, so we're already in the repo directory
    setup_cmd = """
set -e
echo "=== Setting up environment ==="

# Install VGGT and dependencies
echo "Installing dependencies..."
pip install decord h5py tqdm opencv-python-headless
pip install git+https://github.com/facebookresearch/vggt.git

echo "=== Starting preprocessing ==="
"""
    full_command = setup_cmd + command

    print("=" * 60)
    print("VGGT Preprocessing - Gantry Launcher")
    print("=" * 60)
    print(f"Name:        {beaker_name}")
    print(f"GPUs:        {args.gpus}")
    print(f"Preemptible: {args.preemptible}")
    print(f"Priority:    {args.priority}")
    print(f"Clusters:    {clusters}")
    print(f"Command:     {command}")
    print("=" * 60)

    # Launch via gantry
    ctx = click.Context(gantry_run)
    sys.argv = ["--", "/bin/bash", "-c", full_command]
    ctx.forward(
        gantry_run,
        args=["/bin/bash", "-c", full_command],
        clusters=clusters,
        env_vars=[f"{k}={v}" for k, v in env.items()],
        env_secrets=[f"{k}={v}" for k, v in env_secret.items()],
        **gantry_kwargs,
    )


if __name__ == "__main__":
    main()

