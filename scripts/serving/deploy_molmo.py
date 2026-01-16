#!/usr/bin/env python3
"""
Automated deployment script for Molmo models on Modal.

This script automates the entire workflow:
1. Convert Molmo checkpoint to HuggingFace format
2. Upload checkpoint to Modal volume
3. Generate deployment script with correct paths/names
4. Deploy to Modal

Usage:
    python deploy_molmo.py --checkpoint-dir /path/to/molmo/checkpoint --model-name my-model-v1

Example:
    python deploy_molmo.py --checkpoint-dir ./checkpoints/molmo_imdb_final --model-name molmo-imdb-final-v2
"""

import argparse
import os
import sys
import subprocess
import tempfile
import shutil
import re
from pathlib import Path
from typing import Optional


def run_command(cmd: list, cwd: Optional[str] = None, capture_output: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """
    Run a shell command.
    If check is True, CalledProcessError will be raised for non-zero exit codes.
    """
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=check,  # subprocess.run will raise CalledProcessError if check is True and command fails
        capture_output=capture_output,
        text=True
    )
    if capture_output and result.stdout and result.stdout.strip(): # Print if there's actual stdout
        print(f"Output: {result.stdout.strip()}")
    # If check=True and the command fails, CalledProcessError is raised by subprocess.run.
    # The caller is responsible for handling it.
    return result


def sanitize_name(name: str) -> tuple[str, str]:
    """
    Sanitize model name for different contexts.
    Returns (app_name, app_label) where:
    - app_name: allows underscores, used for Python variables
    - app_label: only alphanumerics and dashes, used for Modal labels
    """
    # For app_name: replace dots and dashes with underscores, keep alphanumerics
    app_name_sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    app_name_sanitized = re.sub(r'_+', '_', app_name_sanitized)  # collapse multiple underscores
    app_name_sanitized = app_name_sanitized.strip('_')  # remove leading/trailing underscores

    # For app_label: convert to lowercase, then replace dots and underscores with dashes, keep only lowercase alphanumerics and dashes
    app_label_sanitized = name.lower() # Convert to lowercase first
    app_label_sanitized = re.sub(r'[^a-z0-9-]', '-', app_label_sanitized) # Allow only lowercase, numbers, dash
    app_label_sanitized = re.sub(r'-+', '-', app_label_sanitized)  # collapse multiple dashes
    app_label_sanitized = app_label_sanitized.strip('-')  # remove leading/trailing dashes
    
    # Ensure app_label is not empty after sanitization, default to "modal-app" if it is
    if not app_label_sanitized:
        app_label_sanitized = "modal-app"
    # Ensure app_name is not empty
    if not app_name_sanitized: # Less likely for app_name to be empty with its rules
        app_name_sanitized = "default_model"


    return app_name_sanitized, app_label_sanitized


def check_prerequisites():
    """Check if required tools are available."""
    print("Checking prerequisites...")

    # # Check if modal is installed and authenticated
    # try:
    #     result = run_command(["modal", "auth", "current"], capture_output=True)
    #     print("✓ Modal CLI is installed and authenticated")
    # except (subprocess.CalledProcessError, FileNotFoundError):
    #     print("✗ Modal CLI not found or not authenticated")
    #     print("Please install modal and run 'modal auth new'")
    #     sys.exit(1)

    # Check if we're in the right directory (has mm_olmo structure)
    if not Path("scripts/hf_molmo").exists():
        print("✗ Could not find scripts/hf_molmo directory")
        print("Please run this script from the mm_olmo repository root")
        sys.exit(1)

    print("✓ All prerequisites met")


def convert_checkpoint(checkpoint_dir: str, output_dir: str):
    """Convert Molmo checkpoint to HuggingFace format."""
    print(f"\nConverting checkpoint from {checkpoint_dir} to {output_dir}")

    if not Path(checkpoint_dir).exists():
        print(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # The module path for the conversion script
    # Assumes 'scripts' is a top-level package directory
    module_path = "scripts.hf_molmo.convert_molmo_to_hf"
    
    # Check if the script file actually exists, though we're running as a module
    # This is more of a sanity check for the path derivation
    conversion_script_file = Path(module_path.replace('.', '/') + ".py")
    if not conversion_script_file.exists():
        print(f"✗ Conversion script file not found at expected location: {conversion_script_file}")
        print("Please ensure the path and package structure are correct.")
        sys.exit(1)

    run_command([
        "uv", "run", "python", "-m", module_path,
        checkpoint_dir,
        output_dir
    ])

    print("✓ Checkpoint conversion completed")


def upload_to_modal(hf_checkpoint_dir: str, volume_name: str, remote_path: str):
    """Upload HF checkpoint to Modal volume."""
    print(f"\nUploading {hf_checkpoint_dir} to modal volume {volume_name}:{remote_path}")

    # Check if volume exists, create if not
    try:
        run_command(["modal", "volume", "ls", volume_name], capture_output=True)
        print(f"✓ Volume {volume_name} exists")
    except subprocess.CalledProcessError:
        print(f"Creating volume {volume_name}")
        run_command(["modal", "volume", "create", volume_name])

    # Upload checkpoint
    run_command([
        "modal", "volume", "put",
        volume_name,
        hf_checkpoint_dir,
        remote_path
    ])

    print("✓ Upload completed")


def generate_deployment_script(
        template_script_path: str,
        output_script: str,
        model_name: str,
        app_name: str,
        app_label: str,
        remote_model_path: str,
        original_checkpoint_path: str
):
    """Generate deployment script from template with substituted values."""
    print(f"\nGenerating deployment script: {output_script}")

    template_path = Path(template_script_path)
    if not template_path.exists():
        print(f"✗ Template script not found: {template_path}")
        sys.exit(1)

    with open(template_path, 'r') as f:
        content = f.read()

    # Replace template values
    replacements = {
        'MODEL_NAME = "REPLACEME_USER_MODEL_NAME"': f'MODEL_NAME = "{model_name}"',
        'MODELS_DIR = f"{VOLUME_DIR}/REPLACEME_REMOTE_MODEL_PATH_ON_VOLUME"': f'MODELS_DIR = f"{{VOLUME_DIR}}/{remote_model_path}"',
        'APP_NAME = "REPLACEME_SANITIZED_APP_NAME"': f'APP_NAME = "{app_name}"',
        'APP_LABEL = "REPLACEME_SANITIZED_APP_LABEL"': f'APP_LABEL = "{app_label}"',
        'ORIGINAL_CHECKPOINT_PATH = "REPLACEME_ORIGINAL_CHECKPOINT_PATH"': f'ORIGINAL_CHECKPOINT_PATH = "{original_checkpoint_path}"'
    }

    for old, new in replacements.items():
        if old not in content:
            print(f"Warning: Placeholder '{old}' not found in template {template_script_path}")
        content = content.replace(old, new)

    # Write new script
    with open(output_script, 'w') as f:
        f.write(content)

    print(f"✓ Generated deployment script: {output_script}")


def deploy_to_modal(script_path: str):
    """Deploy the generated script to Modal."""
    print(f"\nDeploying {script_path} to Modal...")

    run_command(["modal", "deploy", script_path])

    print("✓ Deployment completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Automated deployment of Molmo models to Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy with default volume and paths
  python deploy_molmo.py --checkpoint-dir ./checkpoints/molmo_final --model-name molmo-final-v1

  # Deploy with custom volume and keep temporary files
  python deploy_molmo.py --checkpoint-dir /data/molmo_checkpoint --model-name my-molmo-model --volume my-volume --keep-temp
        """
    )

    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the original Molmo checkpoint directory"
    )

    parser.add_argument(
        "--model-name",
        required=True,
        help="Name for the model (will be sanitized for different contexts)"
    )

    parser.add_argument(
        "--volume",
        default="robo-molmo",
        help="Modal volume name (default: robo-molmo)"
    )

    parser.add_argument(
        "--template-script",
        default="modal_base_deployment_script.py",
        help="Path to template deployment script (relative to this script if not absolute)"
    )

    parser.add_argument(
        "--output-dir",
        help="Directory for temporary files (default: creates temp dir)"
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files after deployment"
    )

    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip checkpoint conversion (assumes HF format already exists)"
    )

    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload to Modal (assumes checkpoint already uploaded)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )

    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - showing what would be done:")

    # Check prerequisites
    if not args.dry_run:
        check_prerequisites()

    # Sanitize model name
    app_name, app_label = sanitize_name(args.model_name)
    print(f"\nModel name: {args.model_name}")
    print(f"App name: {app_name}")
    print(f"App label: {app_label}")

    # Set up working directories
    if args.output_dir:
        work_dir = Path(args.output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix=f"molmo_deploy_{app_name}_"))
        temp_cleanup = not args.keep_temp

    hf_checkpoint_dir = work_dir / "hf_checkpoint"
    script_output_path = work_dir / f"{app_name}.py"

    print(f"\nWorking directory: {work_dir}")

    # Determine the remote path for the HF checkpoint on Modal volume
    remote_path_on_volume = f"{app_name}_hf"

    try:
        # Check if model is already uploaded, only if not in dry run
        if not args.dry_run and not args.skip_upload:
            try:
                print(f"Checking if model already exists at {args.volume}:{remote_path_on_volume}...")
                # run_command will raise CalledProcessError if the command fails and check=True (default)
                run_command(["modal", "volume", "ls", args.volume, remote_path_on_volume], capture_output=True)
                # If the above line succeeds (exit code 0), it means the path *does* exist.
                print(f"✓ Model folder appears to be already uploaded at {args.volume}:{remote_path_on_volume}")
                if not args.skip_conversion:
                    print("Skipping checkpoint conversion as model already appears on volume.")
                    args.skip_conversion = True
                print("Skipping upload as model already appears on volume.")
                args.skip_upload = True
            except subprocess.CalledProcessError as e:
                # This block is entered if 'modal volume ls' returns a non-zero exit code.
                if e.returncode == 2: # Exit code 2 from 'modal volume ls <vol> <path>' means path not found.
                    print(f"Model path not found at {args.volume}:{remote_path_on_volume}. Proceeding with conversion/upload.")
                else:
                    # A different error occurred with 'modal volume ls'
                    print(f"Warning: 'modal volume ls {args.volume} {remote_path_on_volume}' command failed with exit code {e.returncode}.")
                    if e.stderr and e.stderr.strip():
                        print(f"Error output from modal: {e.stderr.strip()}")
                    print("Assuming model is not on volume and proceeding with conversion/upload.")
            except FileNotFoundError: # Should be caught by prerequisites, but good to have
                print("✗ Modal CLI not found. Cannot check if model is uploaded. Proceeding with normal workflow.")


        # Step 1: Convert checkpoint
        if not args.skip_conversion:
            if args.dry_run:
                print(f"[DRY RUN] Would convert {args.checkpoint_dir} to {hf_checkpoint_dir}")
            else:
                convert_checkpoint(args.checkpoint_dir, str(hf_checkpoint_dir))
        else:
            # Only print "Skipping" if it wasn't already part of the "already uploaded" message
            if not (not args.dry_run and Path(hf_checkpoint_dir).exists()): # A bit complex, simplify
                 print("Skipping checkpoint conversion (either due to existing upload or user flag).")


        # Step 2: Upload to Modal
        if not args.skip_upload:
            if args.dry_run:
                print(f"[DRY RUN] Would upload {hf_checkpoint_dir} to {args.volume}:{remote_path_on_volume}")
            else:
                if not Path(hf_checkpoint_dir).exists():
                    print(f"✗ HF Checkpoint directory {hf_checkpoint_dir} not found, but conversion was skipped.")
                    print("This can happen if --skip-conversion was used without the files being present, or if an error occurred.")
                    print("Please ensure the HF checkpoint is present at the expected location or do not skip conversion.")
                    sys.exit(1)
                upload_to_modal(str(hf_checkpoint_dir), args.volume, remote_path_on_volume)
        else:
            # Only print "Skipping" if it wasn't already part of the "already uploaded" message
            print("Skipping upload to Modal (either due to existing upload or user flag).")


        # Step 3: Generate deployment script
        template_script_to_use = Path(args.template_script)
        if not template_script_to_use.is_absolute():
            # Assuming deploy_molmo.py is in scripts/serving, and template is also there or path is relative from repo root
            # For simplicity, let's assume it's relative to current working directory or an absolute path
            # Or, more robustly, relative to this script's directory
            # Or, more robustly, relative to this script's directory
            script_dir = Path(__file__).parent
            template_script_to_use = script_dir / args.template_script
            # If the user provides a path like "scripts/serving/modal_base_deployment_script.py" from repo root,
            # and runs deploy_molmo.py from repo root, Path(args.template_script) would be correct.
            # The default "modal_base_deployment_script.py" will be resolved relative to script_dir.
            # If args.template_script is an absolute path, Path() handles it.
            # If it's a relative path, this makes it relative to this script's location.
            # However, the original argparse default implies it might be run from a different CWD.
            # The simplest is to let Path handle it and document that relative paths for --template-script
            # are relative to the CWD, or use the check within generate_deployment_script.
            # For the default, making it relative to the script is safer.
            if args.template_script == parser.get_default("template_script"): # Only adjust default path relative to script
                 template_script_to_use = Path(__file__).parent / args.template_script
            else: # User provided path, assume relative to CWD or absolute
                 template_script_to_use = Path(args.template_script)


        if args.dry_run:
            print(f"[DRY RUN] Would generate deployment script at {script_output_path} from template {template_script_to_use}")
        else:
            generate_deployment_script(
                str(template_script_to_use),
                str(script_output_path),
                args.model_name,       # user-provided model name
                app_name,              # sanitized for python/modal app
                app_label,             # sanitized for modal label
                remote_path_on_volume, # path on modal volume, e.g., app_name_hf
                args.checkpoint_dir    # original molmo checkpoint path
            )

        # Step 4: Deploy to Modal
        if args.dry_run:
            print(f"[DRY RUN] Would deploy {script_output_path} to Modal")
        else:
            deploy_to_modal(str(script_output_path))

        if not args.dry_run:
            print(f"\n🎉 Successfully deployed {args.model_name} to Modal!")
            print(f"Deployment script saved to: {script_output_path}")

            if not args.keep_temp and temp_cleanup:
                print(f"Cleaning up temporary directory: {work_dir}")

    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nDeployment failed: {e}")
        sys.exit(1)
    finally:
        if temp_cleanup and work_dir.exists():
            shutil.rmtree(work_dir)


if __name__ == "__main__":
    main()