from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate TSV files with one shared header.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    first = True
    for input_path in args.inputs:
        df = pd.read_csv(input_path, sep="\t")
        df.to_csv(output, sep="\t", index=False, mode="w" if first else "a", header=first)
        total += len(df)
        first = False
        print(f"Added {len(df)} rows from {input_path}")
    print(f"Saved {total} rows to {output}")


if __name__ == "__main__":
    main()
