"""
Experiment-only script: runs ONLY the BFTS experiment stage.
No plots, no writeup, no review. Use run_post_experiment.py for those.

Usage:
  # Basic: run idea #1 from a JSON file
  python run_experiment.py \
    --load_ideas "ai_scientist/ideas/new_idea_test.json" \
    --idea_idx 1

  # With existing baseline code as starting point
  python run_experiment.py \
    --load_ideas "ai_scientist/ideas/new_idea_test.json" \
    --idea_idx 1 \
    --load_code

  # Then later, run post-experiment separately:
  python run_post_experiment.py \
    --experiment_dir "experiments/2026-.../" \
    --model gpt-5.4 --writeup-type icbinb
"""

import os
import sys
import json
import shutil
import argparse
import os.path as osp
from datetime import datetime

os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))

import torch
from ai_scientist.treesearch.perform_experiments_bfts_with_agentmanager import (
    perform_experiments_bfts,
)
from ai_scientist.treesearch.bfts_utils import (
    idea_to_markdown,
    edit_bfts_config_file,
)
from ai_scientist.utils.token_tracker import token_tracker


def save_token_tracker(idea_dir):
    with open(osp.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(osp.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def get_available_gpus():
    return list(range(torch.cuda.device_count()))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run ONLY the BFTS experiment (no writeup/review)"
    )
    parser.add_argument(
        "--load_ideas",
        type=str,
        required=True,
        help="Path to a JSON file containing pregenerated ideas",
    )
    parser.add_argument(
        "--idea_idx",
        type=int,
        default=0,
        help="Index of the idea to run",
    )
    parser.add_argument(
        "--load_code",
        action="store_true",
        help="Load a .py file with same name as ideas JSON as starting code",
    )
    parser.add_argument(
        "--code_file",
        type=str,
        default=None,
        help="Explicit path to starting code file (overrides --load_code auto-detection)",
    )
    parser.add_argument(
        "--add_dataset_ref",
        action="store_true",
        help="Add HF dataset reference to the idea",
    )
    parser.add_argument(
        "--attempt_id",
        type=int,
        default=0,
        help="Attempt ID for parallel runs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="bfts_config.yaml",
        help="Path to bfts_config.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    available_gpus = get_available_gpus()
    print(f"Using GPUs: {available_gpus}")

    # Load ideas
    with open(args.load_ideas, "r") as f:
        ideas = json.load(f)
    print(f"Loaded {len(ideas)} ideas from {args.load_ideas}")

    idea = ideas[args.idea_idx]
    print(f"Running idea #{args.idea_idx}: {idea.get('Name', 'unnamed')}")
    print(f"  Title: {idea.get('Title', 'N/A')}")

    # Create experiment directory
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    idea_dir = f"experiments/{date}_{idea['Name']}_attempt_{args.attempt_id}"
    os.makedirs(idea_dir, exist_ok=True)
    print(f"Experiment dir: {idea_dir}")

    # Convert idea to markdown
    idea_path_md = osp.join(idea_dir, "idea.md")

    # Load starting code
    code = None
    code_path = None
    if args.code_file:
        # Explicit code file
        if os.path.exists(args.code_file):
            with open(args.code_file, "r") as f:
                code = f.read()
            code_path = args.code_file
            print(f"Loaded starting code from: {args.code_file}")
        else:
            print(f"WARNING: Code file {args.code_file} not found")
    elif args.load_code:
        # Auto-detect: same name as JSON but .py
        auto_code_path = args.load_ideas.rsplit(".", 1)[0] + ".py"
        if os.path.exists(auto_code_path):
            with open(auto_code_path, "r") as f:
                code = f.read()
            code_path = auto_code_path
            print(f"Loaded starting code from: {auto_code_path}")
        else:
            print(f"WARNING: Code file {auto_code_path} not found")

    idea_to_markdown(idea, idea_path_md, code_path)

    # Handle dataset reference
    dataset_ref_code = None
    if args.add_dataset_ref:
        dataset_ref_path = "hf_dataset_reference.py"
        if os.path.exists(dataset_ref_path):
            with open(dataset_ref_path, "r") as f:
                dataset_ref_code = f.read()

    if dataset_ref_code and code:
        added_code = dataset_ref_code + "\n" + code
    elif dataset_ref_code:
        added_code = dataset_ref_code
    elif code:
        added_code = code
    else:
        added_code = None

    if added_code:
        ideas[args.idea_idx]["Code"] = added_code

    # Save idea JSON
    idea_path_json = osp.join(idea_dir, "idea.json")
    with open(idea_path_json, "w") as f:
        json.dump(ideas[args.idea_idx], f, indent=4)

    # Edit config and run experiment
    idea_config_path = edit_bfts_config_file(args.config, idea_dir, idea_path_json)

    print(f"\n{'='*60}")
    print("Starting BFTS Experiment")
    print(f"{'='*60}\n")

    perform_experiments_bfts(idea_config_path)

    # Copy experiment results to top level for easy access
    experiment_results_dir = osp.join(idea_dir, "logs/0-run/experiment_results")
    if os.path.exists(experiment_results_dir):
        dest = osp.join(idea_dir, "experiment_results")
        shutil.copytree(experiment_results_dir, dest, dirs_exist_ok=True)
        print(f"Experiment results copied to: {dest}")

    save_token_tracker(idea_dir)

    print(f"\n{'='*60}")
    print("Experiment completed!")
    print(f"Results saved in: {idea_dir}")
    print(f"\nTo generate paper, run:")
    print(f'  python run_post_experiment.py --experiment_dir "{idea_dir}" --model gpt-5.4 --writeup-type icbinb')
    print(f"{'='*60}")

    # Cleanup child processes
    try:
        import psutil
        import signal
        current = psutil.Process()
        for child in current.children(recursive=True):
            try:
                child.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        gone, alive = psutil.wait_procs(current.children(recursive=True), timeout=3)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        pass
