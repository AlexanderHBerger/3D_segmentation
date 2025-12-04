#!/usr/bin/env python
"""
Test script to verify the resume workflow works correctly.
This performs a series of checks before actually resuming runs.

Usage:
    python test_resume_workflow.py <SWEEP_ID>
    python test_resume_workflow.py username/softmax_grad/sweep_abc123
"""

import sys
import argparse
from pathlib import Path
import wandb


def check_wandb_connection():
    """Test W&B API connection"""
    print("\n" + "="*80)
    print("Test 1: W&B API Connection")
    print("="*80)
    
    try:
        api = wandb.Api()
        print("✓ W&B API initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize W&B API: {e}")
        print("  Make sure WANDB_API_KEY is set correctly")
        return False


def check_sweep_access(sweep_id):
    """Test access to sweep"""
    print("\n" + "="*80)
    print("Test 2: Sweep Access")
    print("="*80)
    
    try:
        api = wandb.Api()
        parts = sweep_id.split('/')
        if len(parts) != 3:
            print(f"✗ Invalid sweep ID format: {sweep_id}")
            print("  Expected format: entity/project/sweep_id")
            return False
        
        entity, project, sweep_name = parts
        sweep_path = f"{entity}/{project}/{sweep_name}"
        sweep = api.sweep(sweep_path)
        
        print(f"✓ Sweep found: {sweep.name}")
        print(f"  State: {sweep.state}")
        print(f"  Total runs: {len(sweep.runs)}")
        return True
    except Exception as e:
        print(f"✗ Failed to access sweep: {e}")
        return False


def check_run_sample(sweep_id):
    """Check a sample run from the sweep"""
    print("\n" + "="*80)
    print("Test 3: Sample Run Check")
    print("="*80)
    
    try:
        api = wandb.Api()
        parts = sweep_id.split('/')
        entity, project, sweep_name = parts
        sweep_path = f"{entity}/{project}/{sweep_name}"
        sweep = api.sweep(sweep_path)
        
        if len(sweep.runs) == 0:
            print("✗ No runs found in sweep")
            return False
        
        # Check first run
        run = sweep.runs[0]
        print(f"Sample Run: {run.id} ({run.name})")
        print(f"  State: {run.state}")
        
        # Check epoch info
        epoch = None
        for key in ['epoch', '_step', 'epochs']:
            if key in run.summary:
                epoch = run.summary[key]
                break
        
        if epoch is None:
            print("  ⚠ Warning: Cannot determine epoch count")
        else:
            print(f"  Epochs: {epoch}")
        
        # Check fold info
        fold = run.config.get('fold', None)
        if fold is None:
            print("  ⚠ Warning: Cannot determine fold")
        else:
            print(f"  Fold: {fold}")
        
        print("✓ Run information accessible")
        return True
    except Exception as e:
        print(f"✗ Failed to access run: {e}")
        return False


def check_checkpoint_structure():
    """Check checkpoint directory structure"""
    print("\n" + "="*80)
    print("Test 4: Checkpoint Directory Structure")
    print("="*80)
    
    base_dir = Path("/ministorage/ahb/scratch/segmentation_model/experiments")
    
    if not base_dir.exists():
        print(f"✗ Base experiments directory not found: {base_dir}")
        return False
    
    print(f"✓ Base directory exists: {base_dir}")
    
    sweeps_dir = base_dir / "sweeps"
    if not sweeps_dir.exists():
        print(f"  ⚠ Warning: Sweeps directory not found: {sweeps_dir}")
        print("    This is OK if no sweep runs have been completed yet")
    else:
        print(f"✓ Sweeps directory exists: {sweeps_dir}")
        
        # Count checkpoint directories
        checkpoint_dirs = list(sweeps_dir.glob("fold_*_*/checkpoint.pth"))
        print(f"  Found {len(checkpoint_dirs)} checkpoint files")
        
        if len(checkpoint_dirs) > 0:
            # Show sample
            sample = checkpoint_dirs[0]
            print(f"  Sample: {sample.parent.name}/checkpoint.pth")
    
    return True


def check_scripts_exist():
    """Check that all required scripts exist"""
    print("\n" + "="*80)
    print("Test 5: Required Scripts")
    print("="*80)
    
    script_dir = Path("/ministorage/ahb/scratch/segmentation_model")
    required_scripts = [
        "get_sweep_runs.py",
        "resume_runs.py",
        "resume_runs.sbatch",
        "submit_resume.sh",
        "train.py"
    ]
    
    all_exist = True
    for script in required_scripts:
        script_path = script_dir / script
        if script_path.exists():
            print(f"✓ {script}")
            # Check if executable (for .py and .sh files)
            if script.endswith(('.py', '.sh')):
                if script_path.stat().st_mode & 0o111:
                    print(f"    (executable)")
                else:
                    print(f"    ⚠ Warning: not executable (run: chmod +x {script})")
        else:
            print(f"✗ {script} not found")
            all_exist = False
    
    return all_exist


def test_get_sweep_runs(sweep_id):
    """Test the get_sweep_runs.py script"""
    print("\n" + "="*80)
    print("Test 6: Get Sweep Runs Script")
    print("="*80)
    
    script_path = Path("/ministorage/ahb/scratch/segmentation_model/get_sweep_runs.py")
    
    if not script_path.exists():
        print(f"✗ Script not found: {script_path}")
        return False
    
    print("Running get_sweep_runs.py (display mode)...")
    print(f"Command: python {script_path} {sweep_id}")
    print("-" * 80)
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", str(script_path), sweep_id],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("-" * 80)
            print("✓ Script executed successfully")
            return True
        else:
            print("-" * 80)
            print(f"✗ Script failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Failed to run script: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test the resume workflow components',
        epilog='Example: python test_resume_workflow.py username/softmax_grad/sweep_abc123'
    )
    parser.add_argument(
        'sweep_id',
        help='W&B sweep ID (format: entity/project/sweep_id)'
    )
    parser.add_argument(
        '--skip-sweep-query',
        action='store_true',
        help='Skip the final sweep query test (faster)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Resume Workflow Test Suite")
    print("="*80)
    print(f"Sweep ID: {args.sweep_id}")
    
    # Run tests
    results = {}
    results['wandb'] = check_wandb_connection()
    results['sweep_access'] = check_sweep_access(args.sweep_id)
    results['run_sample'] = check_run_sample(args.sweep_id)
    results['checkpoint_structure'] = check_checkpoint_structure()
    results['scripts'] = check_scripts_exist()
    
    if not args.skip_sweep_query:
        results['get_sweep_runs'] = test_get_sweep_runs(args.sweep_id)
    
    # Print summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("="*80)
    if all_passed:
        print("✓ All tests passed!")
        print("\nYou can now proceed with the resume workflow:")
        print(f"  1. python get_sweep_runs.py {args.sweep_id} --output runs.txt")
        print(f"  2. ./submit_resume.sh runs.txt 400 --dry-run")
        print(f"  3. ./submit_resume.sh runs.txt 400")
    else:
        print("✗ Some tests failed. Please fix the issues above before proceeding.")
        sys.exit(1)
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
