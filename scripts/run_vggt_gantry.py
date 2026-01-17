#!/usr/bin/env python
"""
Gantry launcher for VGGT depth preprocessing.

Each job uses 1 GPU by default, so you can launch many parallel jobs to process
different house ranges efficiently.

NOTE: By default, git sync is DISABLED. Use --sync to push code changes.

Usage:
    # Process houses 0-1000 with ALIGNED point clouds:
    python scripts/run_vggt_gantry.py \
        "python scripts/precompute_vggt_aligned.py --input_dir /weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor/ObjectNavType/train --output_dir /weka/prior/mattw/data/robot_point_clouds_aligned/ObjectNavType/train --start_house 0 --end_house 1000 --resume" \
        --name vggt-aligned-0

    # If you changed code and need to sync:
    python scripts/run_vggt_gantry.py "..." --name vggt-aligned-0 --sync
"""

import argparse
import os
import subprocess
import sys

import click
from gantry.commands import run as gantry_run


# ============================================================================
# CONFIGURATION
# ============================================================================
SOURCE_REPO = "/weka/prior/mattw/robo_mm_olmo"
GANTRY_REPO = "/weka/prior/mattw/robo_mm_olmo_gantry"
GITHUB_TOKEN_SECRET = "MATTW_GITHUB_TOKEN"
# ============================================================================


def sync_code_to_gantry_repo(job_name: str):
    """Sync code to gantry repo and push to GitHub."""
    print("Syncing code to gantry repo...")
    
    if not os.path.exists(GANTRY_REPO):
        print(f"ERROR: Gantry repo not found at {GANTRY_REPO}")
        sys.exit(1)
    
    rsync_cmd = (
        f"rsync -rzv --delete "
        f"--exclude .git --exclude .gitmodules --exclude .idea "
        f"--exclude '*.html' --exclude '*.pyc' --exclude __pycache__ "
        f"--exclude .cache --exclude .pytest_cache --exclude scratch "
        f"--exclude '*.ipynb' --exclude .venv --exclude uv.lock "
        f"{SOURCE_REPO}/ {GANTRY_REPO}/"
    )
    subprocess.call(rsync_cmd, shell=True)
    
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
    parser.add_argument("command", help="Python command to run")
    parser.add_argument("--name", required=True, help="Beaker job name")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--preemptible", action="store_true", help="Use preemptible instances")
    parser.add_argument("--priority", default="high", choices=["low", "normal", "high", "urgent"])
    parser.add_argument("--cluster", default=None, help="Specific cluster")
    # NOTE: sync is OFF by default - use --sync to push code changes
    parser.add_argument("--sync", action="store_true", help="Sync code to gantry repo (default: OFF)")
    args = parser.parse_args()

    command = args.command
    beaker_name = args.name

    # Only sync if explicitly requested
    if args.sync:
        sync_code_to_gantry_repo(beaker_name)
    
    # Gantry must run from the gantry repo directory
    os.chdir(GANTRY_REPO)

    gantry_kwargs = dict(
        name=beaker_name,
        task_name=beaker_name,
        priority=args.priority,
        budget="ai2/oe-mm",
        gpus=args.gpus,
        shared_memory="16GiB",
        beaker_image="mattw/mattw-vggt-preprocess",
        workspace="ai2/robo-molmo",
        gh_token_secret=GITHUB_TOKEN_SECRET,
        description=f"VGGT preprocessing: {command[:100]}...",
        preemptible=args.preemptible,
    )

    env = dict(
        OMP_NUM_THREADS="8",
        WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000",
        WEKA_PROFILE="weka",
    )

    env_secret = dict(
        BEAKER_TOKEN="MATTW_BEAKER_TOKEN",
    )

    if args.cluster:
        clusters = [args.cluster]
    else:
        clusters = ["ai2/saturn", "ai2/neptune", "ai2/rhea"]

    gantry_kwargs["weka"] = [
        "oe-training-default:/weka/oe-training-default",
        "prior-default:/weka/prior",
    ]

    setup_cmd = """
set -e
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
    print(f"Sync:        {args.sync}")
    print(f"Command:     {command}")
    print("=" * 60)

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
