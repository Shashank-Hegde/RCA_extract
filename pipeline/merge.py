#!/usr/bin/env python

"""
Merge all train.jsonl / val.jsonl files into a single directory.

Usage:
  python pipeline/merge_datasets.py \
    --dirs data/synth data/synth_balanced data/synth_balanced2 \
    --out data/merged
"""

import argparse, json, pathlib

def merge_jsonl_files(dirs, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = open(out_dir / "train.jsonl", "w")
    val_out   = open(out_dir / "val.jsonl", "w")

    for d in dirs:
        train_path = pathlib.Path(d) / "train.jsonl"
        val_path   = pathlib.Path(d) / "val.jsonl"

        if train_path.exists():
            for line in open(train_path):
                train_out.write(line)

        if val_path.exists():
            for line in open(val_path):
                val_out.write(line)

    print(f"✅ Merged {len(dirs)} datasets into → {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    merge_jsonl_files(args.dirs, args.out)
