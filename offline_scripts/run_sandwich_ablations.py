#!/usr/bin/env python3
"""
Batch submit all MLP-Sandwich STU ablation experiments.

This script submits all configs in the sandwich-stu-ablations directory
to SLURM for training.

Usage:
    python offline_scripts/run_sandwich_ablations.py [--dry-run] [--gpus N]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit MLP-Sandwich STU ablation experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without actually submitting",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs per job (default: 1)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="72:00:00",
        help="Max runtime per job (default: 72:00:00)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Specific config files to run (default: all configs in sandwich-stu-ablations/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get the OLMo directory
    script_dir = Path(__file__).parent
    olmo_dir = script_dir.parent
    config_dir = olmo_dir / "configs" / "sandwich-stu-ablations"
    
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        sys.exit(1)
    
    # Find all YAML configs
    if args.configs:
        # Use specific configs provided by user
        config_files = [config_dir / f for f in args.configs if f.endswith('.yaml')]
    else:
        # Use all configs in the directory
        config_files = sorted(config_dir.glob("*.yaml"))
    
    if not config_files:
        print(f"Error: No YAML config files found in {config_dir}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"MLP-Sandwich STU Ablation Experiment Submission")
    print(f"{'='*80}")
    print(f"Config directory: {config_dir}")
    print(f"Number of experiments: {len(config_files)}")
    print(f"GPUs per job: {args.gpus}")
    print(f"Max time per job: {args.time}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'SUBMIT'}")
    print(f"{'='*80}\n")
    
    # List all experiments
    print("Experiments to run:")
    for i, config_file in enumerate(config_files, 1):
        print(f"  {i}. {config_file.name}")
    print()
    
    if args.dry_run:
        print("Dry run mode - no jobs will be submitted.")
        return
    
    # Confirm submission
    try:
        response = input("Submit all jobs? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted.")
            return
    except KeyboardInterrupt:
        print("\nAborted.")
        return
    
    print(f"\n{'='*80}")
    print("Submitting jobs...")
    print(f"{'='*80}\n")
    
    # Submit each config
    submitted = 0
    failed = 0
    
    for config_file in config_files:
        print(f"Submitting: {config_file.name} ...", end=" ", flush=True)
        
        cmd = [
            "python",
            str(script_dir / "run_slurm_job_config.py"),
            "--config", str(config_file),
            "--gpus", str(args.gpus),
            "--time", args.time,
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=olmo_dir,
        )
        
        if result.returncode == 0:
            # Extract job ID from output
            job_id = None
            for line in result.stdout.split('\n'):
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    break
            
            if job_id:
                print(f"✓ (Job ID: {job_id})")
            else:
                print("✓")
            submitted += 1
        else:
            print("✗")
            print(f"  Error: {result.stderr}")
            failed += 1
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Successfully submitted: {submitted}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(config_files)}")
    print(f"{'='*80}\n")
    
    if submitted > 0:
        print("Monitor jobs with:")
        print("  squeue -u $USER")
        print("\nView logs in:")
        print("  logs/")
        print("\nCheck W&B dashboard:")
        print("  https://wandb.ai/kg4280-princeton-university/olmo-sandwich-stu-ablations")


if __name__ == "__main__":
    main()

