"""
Post-experiment script: runs only the stages AFTER BFTS experiment completion.
  1. Aggregate plots
  2. Write paper (icbinb 4-page or normal 8-page)
  3. LLM + VLM review

Usage:
  python run_post_experiment.py \
    --experiment_dir "experiments/2026-04-04_15-18-44_conflict_memory_allocation_attempt_0" \
    --model gpt-5.4
"""

import os
import sys
import json
import shutil
import argparse
import re
from datetime import datetime

os.environ["AI_SCIENTIST_ROOT"] = os.path.dirname(os.path.abspath(__file__))

from ai_scientist.llm import create_client
from ai_scientist.perform_plotting import aggregate_plots
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_icbinb_writeup import (
    perform_writeup as perform_icbinb_writeup,
    gather_citations,
)
from ai_scientist.perform_llm_review import perform_review, load_paper
from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
from ai_scientist.utils.token_tracker import token_tracker


def find_pdf_path_for_review(idea_dir):
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    reflection_pdfs = [f for f in pdf_files if "reflection" in f]
    if not reflection_pdfs:
        return None
    final_pdfs = [f for f in reflection_pdfs if "final" in f.lower()]
    if final_pdfs:
        return os.path.join(idea_dir, final_pdfs[0])
    reflection_nums = []
    for f in reflection_pdfs:
        match = re.search(r"reflection[_.]?(\d+)", f)
        if match:
            reflection_nums.append((int(match.group(1)), f))
    if reflection_nums:
        highest = max(reflection_nums, key=lambda x: x[0])
        return os.path.join(idea_dir, highest[1])
    return os.path.join(idea_dir, reflection_pdfs[0])


def save_token_tracker(idea_dir):
    with open(os.path.join(idea_dir, "token_tracker.json"), "w") as f:
        json.dump(token_tracker.get_summary(), f)
    with open(os.path.join(idea_dir, "token_tracker_interactions.json"), "w") as f:
        json.dump(token_tracker.get_interactions(), f)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run post-experiment stages (plots, writeup, review)"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory (e.g. experiments/2026-04-04_...)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4",
        help="Model to use for all post-experiment stages",
    )
    parser.add_argument(
        "--writeup-type",
        type=str,
        default="icbinb",
        choices=["normal", "icbinb"],
        help="Type of writeup (normal=8 page, icbinb=4 page)",
    )
    parser.add_argument(
        "--writeup-retries",
        type=int,
        default=3,
        help="Number of writeup attempts",
    )
    parser.add_argument(
        "--num_cite_rounds",
        type=int,
        default=20,
        help="Number of citation rounds",
    )
    parser.add_argument(
        "--model_plots",
        type=str,
        default=None,
        help="Model for plot aggregation (default: same as --model). Use a lighter model if --model causes timeouts.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot aggregation",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Skip writeup",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Skip review",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    idea_dir = args.experiment_dir
    model = args.model

    if not os.path.exists(idea_dir):
        print(f"Error: experiment directory not found: {idea_dir}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Post-Experiment Pipeline")
    print(f"Experiment dir: {idea_dir}")
    print(f"Model: {model}")
    print(f"{'='*60}")

    # ── Step 1: Copy experiment results ──
    experiment_results_dir = os.path.join(idea_dir, "logs/0-run/experiment_results")
    exp_results_dest = os.path.join(idea_dir, "experiment_results")
    if os.path.exists(experiment_results_dir) and not os.path.exists(exp_results_dest):
        print("\n[Step 1] Copying experiment results...")
        shutil.copytree(experiment_results_dir, exp_results_dest, dirs_exist_ok=True)
        print(f"  Copied to {exp_results_dest}")
    elif os.path.exists(exp_results_dest):
        print("\n[Step 1] experiment_results already exists, skipping copy.")
    else:
        print("\n[Step 1] WARNING: No experiment_results found in logs. Plots may fail.")

    # ── Step 2: Aggregate plots ──
    if not args.skip_plots:
        plots_model = args.model_plots or model
        print(f"\n[Step 2] Aggregating plots with model={plots_model}...")
        try:
            aggregate_plots(base_folder=idea_dir, model=plots_model)
            print("  Plot aggregation complete.")
        except Exception as e:
            print(f"  WARNING: Plot aggregation failed: {e}")
    else:
        print("\n[Step 2] Skipping plot aggregation.")

    # Clean up experiment_results copy (match original pipeline behavior)
    if os.path.exists(exp_results_dest):
        shutil.rmtree(exp_results_dest, ignore_errors=True)

    save_token_tracker(idea_dir)

    # ── Step 3: Write paper ──
    if not args.skip_writeup:
        print(f"\n[Step 3] Writing paper ({args.writeup_type}, model={model})...")
        writeup_success = False

        citations_text = gather_citations(
            idea_dir,
            num_cite_rounds=args.num_cite_rounds,
            small_model=model,
        )

        for attempt in range(args.writeup_retries):
            print(f"  Writeup attempt {attempt + 1} of {args.writeup_retries}")
            if args.writeup_type == "normal":
                writeup_success = perform_writeup(
                    base_folder=idea_dir,
                    small_model=model,
                    big_model=model,
                    page_limit=8,
                    citations_text=citations_text,
                )
            else:
                writeup_success = perform_icbinb_writeup(
                    base_folder=idea_dir,
                    small_model=model,
                    big_model=model,
                    page_limit=4,
                    citations_text=citations_text,
                )
            if writeup_success:
                print("  Writeup succeeded!")
                break

        if not writeup_success:
            print("  WARNING: Writeup failed after all retries.")
    else:
        print("\n[Step 3] Skipping writeup.")

    save_token_tracker(idea_dir)

    # ── Step 4: Review paper ──
    if not args.skip_review and not args.skip_writeup:
        print(f"\n[Step 4] Reviewing paper with model={model}...")
        pdf_path = find_pdf_path_for_review(idea_dir)
        if pdf_path and os.path.exists(pdf_path):
            print(f"  Paper found: {pdf_path}")
            paper_content = load_paper(pdf_path)
            client, client_model = create_client(model)
            review_text = perform_review(paper_content, client_model, client)
            review_img_cap_ref = perform_imgs_cap_ref_review(
                client, client_model, pdf_path
            )
            with open(os.path.join(idea_dir, "review_text.txt"), "w") as f:
                f.write(json.dumps(review_text, indent=4))
            with open(os.path.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(review_img_cap_ref, f, indent=4)
            print("  Review completed!")
        else:
            print("  WARNING: No PDF found for review.")
    else:
        print("\n[Step 4] Skipping review.")

    save_token_tracker(idea_dir)
    print(f"\n{'='*60}")
    print("All post-experiment stages completed!")
    print(f"Results saved in: {idea_dir}")
    print(f"{'='*60}")
