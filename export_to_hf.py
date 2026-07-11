#!/usr/bin/env python3
"""
One-time helper: upload the inference-only mansa-nano checkpoint to the Hugging Face Hub.

Uploads exactly the three files needed for inference:
  - model_<step>.pt
  - meta_<step>.json
  - tokenizer.pkl

Usage (must be logged in: `huggingface-cli login`, or set HF_TOKEN):
  python export_to_hf.py \
      --src /local2/sheriff/.cache/nanochat/run_archive/20260711_022812_100-llm-mansa-nano-d24 \
      --repo-id African-Languages-Lab/mansa-nano

By default the source dir is the archived run's sft/ + tokenizer/ layout; we resolve the
newest model_*.pt in <src>/sft and the tokenizer.pkl in <src>/tokenizer.
"""
import argparse
import glob
import os
import sys


def find_last_step(directory):
    files = glob.glob(os.path.join(directory, "model_*.pt"))
    if not files:
        raise FileNotFoundError(f"No model_*.pt found in {directory}")
    return max(int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in files)


def main():
    parser = argparse.ArgumentParser(description="Upload mansa-nano inference weights to Hugging Face")
    parser.add_argument("--src", type=str, required=True,
                        help="Archived run dir containing sft/ and tokenizer/ (or a flat dir with the files)")
    parser.add_argument("--repo-id", type=str, default="African-Languages-Lab/mansa-nano")
    parser.add_argument("--private", action="store_true", help="Create the repo as private (default: public)")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest in sft/)")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    # Resolve the sft checkpoint dir and tokenizer dir, supporting both the archive layout
    # (<src>/sft, <src>/tokenizer) and a flat directory that already holds all files.
    sft_dir = os.path.join(args.src, "sft")
    tok_dir = os.path.join(args.src, "tokenizer")
    if not os.path.isdir(sft_dir):
        sft_dir = args.src
    if not os.path.isdir(tok_dir):
        tok_dir = args.src

    step = args.step if args.step is not None else find_last_step(sft_dir)
    model_path = os.path.join(sft_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(sft_dir, f"meta_{step:06d}.json")
    tokenizer_path = os.path.join(tok_dir, "tokenizer.pkl")

    for path in (model_path, meta_path, tokenizer_path):
        if not os.path.isfile(path):
            sys.exit(f"ERROR: required file missing: {path}")

    api = HfApi()
    print(f"Creating repo {args.repo_id} (private={args.private})...")
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    for path in (model_path, meta_path, tokenizer_path):
        name = os.path.basename(path)
        print(f"Uploading {name} ({os.path.getsize(path) / 1e6:.1f} MB)...")
        api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=args.repo_id, repo_type="model")

    print(f"Done. View at https://huggingface.co/{args.repo_id}")
    print(f"Uploaded step {step}: {os.path.basename(model_path)}, {os.path.basename(meta_path)}, tokenizer.pkl")


if __name__ == "__main__":
    main()
