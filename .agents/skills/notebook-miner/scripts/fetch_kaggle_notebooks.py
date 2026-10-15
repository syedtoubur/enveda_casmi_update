#!/usr/bin/env python3
"""Fetch all public notebooks for a Kaggle competition token-efficiently."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def get_kernels_list(competition: str, page_size: int = 100) -> list[dict]:
    """Retrieve list of kernels using Kaggle CLI."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["kaggle", "kernels", "list", "--competition", competition, "--page-size", str(page_size)]
    r = subprocess.run(cmd, capture_output=True, env=env, errors="replace")
    
    lines = r.stdout.splitlines()
    kernels = []
    # Header format: ref, title, author, lastRunTime, totalVotes
    # First line is header or separator
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("ref") or line_s.startswith("---"):
            continue
        # Split on whitespace, but note title can contain spaces. The ref is the first token: <user>/<slug>
        tokens = line_s.split()
        if not tokens:
            continue
        ref = tokens[0]
        if "/" in ref:
            kernels.append({"ref": ref})

    return kernels


def pull_kernel(ref: str, target_dir: Path) -> bool:
    """Pull kernel notebook into target_dir."""
    slug = ref.split("/")[1]
    kernel_dir = target_dir / slug
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Check if .ipynb already exists
    existing_ipynbs = list(kernel_dir.glob("*.ipynb"))
    if existing_ipynbs:
        return True

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = ["kaggle", "kernels", "pull", ref, "-p", str(kernel_dir)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, errors="replace", timeout=60)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [Timeout] Pulling {ref} timed out.")
        return False
    except Exception as e:
        print(f"  [Error] Failed to pull {ref}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fetch Kaggle public notebooks.")
    parser.add_argument("--competition", default="enveda-CASMI26-molecule-id-mass-spectra", help="Competition slug")
    parser.add_argument("--out-dir", default=".scratch/kaggle_notebooks", help="Output directory")
    parser.add_argument("--limit", type=int, default=50, help="Max number of notebooks to fetch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching public kernels for {args.competition}...")
    kernels = get_kernels_list(args.competition, page_size=args.limit)
    print(f"Found {len(kernels)} candidate kernels.")

    pulled_count = 0
    for idx, k in enumerate(kernels[:args.limit], 1):
        ref = k["ref"]
        print(f"[{idx}/{min(len(kernels), args.limit)}] Pulling {ref}...")
        ok = pull_kernel(ref, out_dir)
        if ok:
            pulled_count += 1
        time.sleep(0.5)

    print(f"\nCompleted! Pulled {pulled_count} notebooks into {out_dir}")


if __name__ == "__main__":
    main()
