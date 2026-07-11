"""
Load a mansa-nano checkpoint for inference, pulling the weights + tokenizer from the
Hugging Face Hub (or a local directory) and building the model.

The HF repo is expected to contain, at its root:
  - model_<step>.pt      (model weights)
  - meta_<step>.json     (model config / metadata)
  - tokenizer.pkl        (pickled tiktoken Encoding)

Nothing else is required for inference (no optimizer shards, no training code).
"""
import os
import logging

from nanochat.checkpoint_manager import build_model, find_last_step

logger = logging.getLogger(__name__)

DEFAULT_HF_REPO = "African-Languages-Lab/mansa-nano"


def resolve_checkpoint_dir(repo_id=DEFAULT_HF_REPO, local_dir=None, revision=None):
    """
    Return a local directory that contains model_*.pt, meta_*.json and tokenizer.pkl.

    If `local_dir` is given, it is used directly (offline / pre-downloaded). Otherwise the
    repo is fetched from the Hugging Face Hub via snapshot_download (cached on disk).
    """
    if local_dir is not None:
        if not os.path.isdir(local_dir):
            raise FileNotFoundError(f"local_dir does not exist: {local_dir}")
        return local_dir

    from huggingface_hub import snapshot_download

    logger.info(f"Fetching mansa-nano weights from Hugging Face repo '{repo_id}'...")
    checkpoint_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["model_*.pt", "meta_*.json", "tokenizer.pkl", "token_bytes.pt"],
    )
    return checkpoint_dir


def load_from_hf(device, phase="eval", repo_id=DEFAULT_HF_REPO, local_dir=None, step=None, revision=None):
    """
    Load (model, tokenizer, meta) for inference.

    Args:
        device: torch.device to load the model onto.
        phase: "eval" (default) or "train".
        repo_id: Hugging Face repo id hosting the weights.
        local_dir: optional local dir with the checkpoint files (skips the HF download).
        step: optional explicit checkpoint step; if None, the largest step on disk is used.
        revision: optional HF revision (branch / tag / commit).
    """
    checkpoint_dir = resolve_checkpoint_dir(repo_id=repo_id, local_dir=local_dir, revision=revision)
    # The tokenizer lives alongside the weights in the same directory.
    os.environ["NANOCHAT_TOKENIZER_DIR"] = checkpoint_dir
    if step is None:
        step = find_last_step(checkpoint_dir)
    logger.info(f"Loading mansa-nano from {checkpoint_dir} (step {step})")
    model, tokenizer, meta = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta
