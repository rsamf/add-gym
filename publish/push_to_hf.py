#!/usr/bin/env python3
"""Download a model checkpoint from S3 and push it to Hugging Face Hub.

Usage:
    python publish/push_to_hf.py \
        --s3-path s3://bucket/outputs/output/abc1234/model.pt \
        --repo-id username/g1-walk

The HF repo is created if it doesn't already exist. Subsequent runs update the
weights in-place.

Requires:
    - AWS credentials configured (env vars, instance profile, etc.)
    - HF_TOKEN environment variable set to a write-access token
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import boto3
import torch
from huggingface_hub import HfApi


def download_from_s3(s3_path: str, local_path: Path) -> Path:
    """Download a single object from S3.

    Args:
        s3_path: Full S3 URI, e.g. s3://bucket/key/model.pt
        local_path: Local directory to download into.

    Returns:
        Path to the downloaded file.
    """
    # Parse s3://bucket/key
    assert s3_path.startswith("s3://"), f"Expected s3:// URI, got: {s3_path}"
    without_scheme = s3_path[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")

    dest = local_path / os.path.basename(key)
    print(f"Downloading s3://{bucket}/{key} → {dest}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(dest))
    return dest


def extract_metadata(checkpoint: dict) -> dict:
    """Pull training metadata out of a checkpoint dict."""
    meta: dict = {}
    if "iter" in checkpoint:
        meta["training_iterations"] = checkpoint["iter"]
    if "sample_count" in checkpoint:
        meta["total_samples"] = checkpoint["sample_count"]
    if "model" in checkpoint:
        state = checkpoint["model"]
        meta["num_parameters"] = sum(
            v.numel() for v in state.values() if isinstance(v, torch.Tensor)
        )
    return meta


def strip_optimizer(checkpoint: dict) -> dict:
    """Return a copy of the checkpoint without the optimizer state."""
    return {k: v for k, v in checkpoint.items() if k != "optimizer"}


def build_model_card(repo_id: str, meta: dict) -> str:
    """Generate a README model card as a string."""
    iters = meta.get("training_iterations", "N/A")
    samples = meta.get("total_samples", "N/A")
    params = meta.get("num_parameters", "N/A")

    iters_str = f"{iters:,}" if isinstance(iters, int) else str(iters)
    samples_str = f"{samples:,}" if isinstance(samples, int) else str(samples)
    params_str = f"{params:,}" if isinstance(params, int) else str(params)

    return f"""\
---
license: mit
library_name: pytorch
tags:
  - reinforcement-learning
  - locomotion
  - robotics
  - g1
---

# {repo_id.split('/')[-1]}

PyTorch checkpoint for a G1 humanoid locomotion policy trained with
ADD (Adversarial Distillation of Demonstrations).

## Checkpoint info

| Key | Value |
|-----|-------|
| Training iterations | `{iters_str}` |
| Total environment samples | `{samples_str}` |
| Number of parameters | `{params_str}` |

## Usage

```python
import torch

checkpoint = torch.load("model.pt", map_location="cpu")
state_dict = checkpoint["model"]
# Load into your agent:
# agent.load_state_dict(state_dict)
```
"""


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoint from S3 and push to Hugging Face Hub"
    )
    parser.add_argument(
        "--s3-path",
        required=True,
        help="S3 URI to the model checkpoint, e.g. s3://bucket/outputs/output/<sha>/model.pt",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo id, e.g. username/g1-walk",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (only applies on first creation)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Download checkpoint from S3
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        ckpt_file = download_from_s3(args.s3_path, tmp)

        # ------------------------------------------------------------------
        # 2. Load & process checkpoint
        # ------------------------------------------------------------------
        print(f"Loading checkpoint from {ckpt_file} ...")
        checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)

        meta = extract_metadata(checkpoint)
        print(f"  iterations : {meta.get('training_iterations', '?')}")
        print(f"  samples    : {meta.get('total_samples', '?')}")
        print(f"  parameters : {meta.get('num_parameters', '?')}")

        print("Stripping optimizer state ...")
        checkpoint = strip_optimizer(checkpoint)

        # Overwrite the file with the stripped version
        torch.save(checkpoint, ckpt_file)

        # Write metadata sidecar
        meta_path = tmp / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        # Write model card
        readme_path = tmp / "README.md"
        readme_path.write_text(build_model_card(args.repo_id, meta))

        # ------------------------------------------------------------------
        # 3. Push to Hugging Face Hub
        # ------------------------------------------------------------------
        api = HfApi()

        print(f"Ensuring repo {args.repo_id} exists ...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )

        print(f"Uploading to https://huggingface.co/{args.repo_id} ...")
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(tmp),
            repo_type="model",
            commit_message=f"Update checkpoint (iter {meta.get('training_iterations', '?')})",
        )

    print(f"\n✅  Done → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
