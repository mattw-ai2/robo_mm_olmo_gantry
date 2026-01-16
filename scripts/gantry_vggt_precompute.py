#!/usr/bin/env python
"""
Gantry launcher for VGGT depth preprocessing.

This script launches beaker jobs to precompute point clouds from the VIDA dataset.
Each job uses multiple GPUs and processes a specific range of houses.

Usage:
    # Dry run to see what jobs would be launched:
    python scripts/gantry_vggt_precompute.py --name vggt-precompute --num_jobs 16 --dry_run

    # Launch 16 parallel jobs (each with 8 GPUs), processing ~900 houses each:
    python scripts/gantry_vggt_precompute.py --name vggt-precompute --num_jobs 16

    # Use preemptible instances (cheaper but can be interrupted - safe with --resume):
    python scripts/gantry_vggt_precompute.py --name vggt-precompute --num_jobs 16 --preemptible

    # Custom GPU count per job:
    python scripts/gantry_vggt_precompute.py --name vggt-precompute --num_jobs 8 --gpus 4

    # Process validation split:
    python scripts/gantry_vggt_precompute.py --name vggt-val --task_type ObjectNavType --split validation
"""

import argparse
import os
import sys
from typing import List, Optional

try:
    import click
    from gantry.commands import run as gantry_run
except ImportError:
    print("Error: gantry not installed. Install with: pip install beaker-gantry")
    sys.exit(1)


# Configuration
DEFAULT_INPUT_BASE = "/weka/prior/datasets/vida_datasets/31Jul2025_timebudget_05hz_FPIN_new_procthor"
DEFAULT_OUTPUT_BASE = "/weka/prior/mattw/data/robot_point_clouds"
DEFAULT_TASK_TYPE = "ObjectNavType"
DEFAULT_SPLIT = "train"
TOTAL_HOUSES_OBJECTNAV_TRAIN = 14418  # Pre-computed count


def get_house_count(input_dir: str) -> int:
    """Count number of house directories in input."""
    if os.path.exists(input_dir):
        return len([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    return 0


def build_precompute_command(
    task_type: str,
    split: str,
    input_base: str,
    output_base: str,
    num_gpus: int = 8,
    start_house: Optional[int] = None,
    end_house: Optional[int] = None,
    resume: bool = True,
) -> str:
    """Build the preprocessing command."""
    input_dir = f"{input_base}/{task_type}/{split}"
    output_dir = f"{output_base}/{task_type}/{split}_v2"
    
    cmd = f"python scripts/precompute_vggt_depth.py --input_dir {input_dir} --output_dir {output_dir} --num_gpus {num_gpus}"
    
    if start_house is not None:
        cmd += f" --start_house {start_house}"
    if end_house is not None:
        cmd += f" --end_house {end_house}"
    if resume:
        cmd += " --resume"
    
    return cmd


def launch_gantry_job(
    name: str,
    command: str,
    priority: str = "normal",
    preemptible: bool = False,
    gpus: int = 8,
    clusters: Optional[List[str]] = None,
    dry_run: bool = False,
):
    """Launch a single gantry job."""
    
    if clusters is None:
        clusters = ["ai2/ceres-cirrascale", "ai2/jupiter-cirrascale-2", "ai2/pluto-cirrascale"]
    
    gantry_kwargs = dict(
        name=name,
        task_name=name,
        priority=priority,
        budget="ai2/oe-training",
        gpus=gpus,
        shared_memory="64GiB",
        venv="base",
        beaker_image="ai2/cuda12.4-ubuntu22.04",
        workspace="ai2/prior",
        conda=False,
        preemptible=preemptible,
        description=f"VGGT preprocessing: {command}",
    )
    
    env = dict(
        OMP_NUM_THREADS="8",
        WEKA_ENDPOINT_URL="https://weka-aus.beaker.org:9000",
        WEKA_PROFILE="weka",
    )
    
    env_secret = dict(
        HF_ACCESS_TOKEN="MATTW_HF_TOKEN",
        BEAKER_TOKEN="MATTW_BEAKER_TOKEN",
    )
    
    # Add weka mounts
    gantry_kwargs["weka"] = ["oe-training-default:/weka/oe-training-default", "prior:/weka/prior"]
    
    # Setup command: install dependencies and run
    setup_cmd = """
set -e
cd /weka/prior/mattw/robo_mm_olmo
pip install decord h5py tqdm opencv-python-headless
pip install git+https://github.com/facebookresearch/vggt.git
"""
    full_command = setup_cmd + command
    
    if dry_run:
        return
    
    # Launch via gantry
    ctx = click.Context(gantry_run)
    sys.argv = ["--", "/bin/bash", "-c", full_command]
    ctx.forward(
        gantry_run,
        arg=["/bin/bash", "-c", full_command],
        cluster=clusters,
        env=[f"{k}={v}" for k, v in env.items()],
        env_secret=[f"{k}={v}" for k, v in env_secret.items()],
        **gantry_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Launch VGGT preprocessing jobs via Gantry/Beaker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Base name for the beaker job(s)"
    )
    parser.add_argument(
        "--task_type", type=str, default=DEFAULT_TASK_TYPE,
        help=f"Task type to process (default: {DEFAULT_TASK_TYPE})"
    )
    parser.add_argument(
        "--split", type=str, default=DEFAULT_SPLIT,
        help=f"Data split to process (default: {DEFAULT_SPLIT})"
    )
    parser.add_argument(
        "--input_base", type=str, default=DEFAULT_INPUT_BASE,
        help="Base input directory"
    )
    parser.add_argument(
        "--output_base", type=str, default=DEFAULT_OUTPUT_BASE,
        help="Base output directory"
    )
    parser.add_argument(
        "--num_jobs", type=int, default=1,
        help="Number of parallel jobs to launch (each processes a subset of houses)"
    )
    parser.add_argument(
        "--gpus", type=int, default=8,
        help="Number of GPUs per job (default: 8)"
    )
    parser.add_argument(
        "--start_house", type=int, default=None,
        help="Start house index (for manual range control)"
    )
    parser.add_argument(
        "--end_house", type=int, default=None,
        help="End house index (for manual range control)"
    )
    parser.add_argument(
        "--priority", type=str, default="normal",
        choices=["low", "normal", "high", "urgent"],
        help="Job priority"
    )
    parser.add_argument(
        "--preemptible", action="store_true",
        help="Use preemptible instances (cheaper but can be interrupted, safe with --resume)"
    )
    parser.add_argument(
        "--no_resume", action="store_true",
        help="Don't use --resume flag (reprocess all houses)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without launching jobs"
    )
    
    args = parser.parse_args()
    
    input_dir = f"{args.input_base}/{args.task_type}/{args.split}"
    
    # Get total house count for splitting
    total_houses = get_house_count(input_dir)
    if total_houses == 0:
        print(f"Warning: Could not count houses in {input_dir}")
        if args.task_type == "ObjectNavType" and args.split == "train":
            total_houses = TOTAL_HOUSES_OBJECTNAV_TRAIN
            print(f"Using cached count: {total_houses}")
        else:
            print("Please specify --start_house and --end_house manually")
            sys.exit(1)
    
    print("=" * 60)
    print("VGGT Preprocessing - Gantry Job Launcher")
    print("=" * 60)
    print(f"Task:         {args.task_type}/{args.split}")
    print(f"Total houses: {total_houses}")
    print(f"Num jobs:     {args.num_jobs}")
    print(f"GPUs/job:     {args.gpus}")
    print(f"Preemptible:  {args.preemptible}")
    print(f"Resume:       {not args.no_resume}")
    print("=" * 60)
    
    # Determine house range
    start = args.start_house if args.start_house is not None else 0
    end = args.end_house if args.end_house is not None else total_houses
    total_to_process = end - start
    
    if args.num_jobs == 1:
        # Single job mode
        cmd = build_precompute_command(
            task_type=args.task_type,
            split=args.split,
            input_base=args.input_base,
            output_base=args.output_base,
            num_gpus=args.gpus,
            start_house=args.start_house,
            end_house=args.end_house,
            resume=not args.no_resume,
        )
        
        print(f"\nJob: {args.name}")
        print(f"Houses: {start} - {end} ({total_to_process} houses)")
        print(f"Command: {cmd}")
        
        if not args.dry_run:
            print("\nLaunching job...")
            launch_gantry_job(
                name=args.name,
                command=cmd,
                priority=args.priority,
                preemptible=args.preemptible,
                gpus=args.gpus,
                dry_run=args.dry_run,
            )
    else:
        # Multi-job mode: split house range across jobs
        houses_per_job = total_to_process // args.num_jobs
        remainder = total_to_process % args.num_jobs
        
        print(f"\nSplitting {total_to_process} houses across {args.num_jobs} jobs")
        print(f"Houses per job: ~{houses_per_job} (+ {remainder} remainder distributed)\n")
        
        jobs = []
        current_start = start
        
        for i in range(args.num_jobs):
            # Distribute remainder across first few jobs
            job_houses = houses_per_job + (1 if i < remainder else 0)
            job_end = current_start + job_houses
            
            job_name = f"{args.name}-{i:02d}"
            cmd = build_precompute_command(
                task_type=args.task_type,
                split=args.split,
                input_base=args.input_base,
                output_base=args.output_base,
                num_gpus=args.gpus,
                start_house=current_start,
                end_house=job_end,
                resume=not args.no_resume,
            )
            
            jobs.append((job_name, cmd, current_start, job_end))
            current_start = job_end
        
        # Print job summary
        print(f"{'Job':<20} {'Houses':<20} {'Count':<10}")
        print("-" * 50)
        for job_name, cmd, job_start, job_end in jobs:
            print(f"{job_name:<20} {job_start:>6} - {job_end:<10} {job_end - job_start:<10}")
        
        if args.dry_run:
            print("\n[DRY RUN] Commands that would be executed:")
            for job_name, cmd, _, _ in jobs:
                print(f"\n{job_name}:")
                print(f"  {cmd}")
        else:
            print(f"\nLaunching {len(jobs)} jobs...")
            for job_name, cmd, _, _ in jobs:
                print(f"  Launching {job_name}...")
                launch_gantry_job(
                    name=job_name,
                    command=cmd,
                    priority=args.priority,
                    preemptible=args.preemptible,
                    gpus=args.gpus,
                    dry_run=args.dry_run,
                )
    
    if args.dry_run:
        print("\n" + "=" * 60)
        print("[DRY RUN] No jobs launched. Remove --dry_run to submit.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Jobs submitted! Monitor with: beaker experiment list")
        print("=" * 60)


if __name__ == "__main__":
    main()
