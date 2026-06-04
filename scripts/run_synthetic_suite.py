#!/usr/bin/env python3
"""Run a synthetic validation suite across multiple scenarios."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from scripts.simulate_superfamily import simulate_dataset
from scripts.evaluate_synthetic import evaluate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic validation suite runner")
    parser.add_argument("--output-dir", default="output/synthetic_suite", help="Output directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch-size", type=int, default=16, help="ESM batch size")
    parser.add_argument("--embed-cache", default=None, help="Embedding cache (npz)")
    parser.add_argument("--amp", action="store_true", help="Use AMP during ESM inference")
    parser.add_argument("--data-parallel", action="store_true", help="Use DataParallel if available")
    parser.add_argument("--mock-embeddings", action="store_true", help="Use random embeddings (fast test)")
    parser.add_argument("--mock-dim", type=int, default=256, help="Embedding dimension for mock mode")
    parser.add_argument("--top-k", type=int, default=30, help="Top-k for recall metric")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    scenarios: List[Dict[str, object]] = [
        {
            "name": "signal_low",
            "length": 250,
            "background_rate": 0.01,
            "hotspot_rate": 0.10,
            "composition": "neutral",
        },
        {
            "name": "signal_mid",
            "length": 250,
            "background_rate": 0.01,
            "hotspot_rate": 0.25,
            "composition": "neutral",
        },
        {
            "name": "signal_high",
            "length": 250,
            "background_rate": 0.01,
            "hotspot_rate": 0.50,
            "composition": "neutral",
        },
        {
            "name": "null",
            "length": 250,
            "background_rate": 0.02,
            "hotspot_rate": 0.02,
            "composition": "neutral",
        },
        {
            "name": "hydrophobic_bias",
            "length": 250,
            "background_rate": 0.01,
            "hotspot_rate": 0.25,
            "composition": "hydrophobic",
        },
    ]

    summary_rows = []
    for scenario in scenarios:
        scenario_dir = output_root / scenario["name"]
        data = simulate_dataset(
            output_dir=scenario_dir,
            families=["FAM1", "FAM2", "FAM3"],
            seqs_per_family=15,
            length=int(scenario["length"]),
            seed=7,
            composition=str(scenario["composition"]),
            motif="HXD",
            motif_positions=None,
            background_rate=float(scenario["background_rate"]),
            hotspot_rate=float(scenario["hotspot_rate"]),
        )

        df = evaluate_dataset(
            fasta_path=data["fasta"],
            labels_path=data["labels"],
            output_dir=scenario_dir,
            device=args.device,
            batch_size=max(1, args.batch_size),
            embed_cache=args.embed_cache,
            use_amp=args.amp,
            use_data_parallel=args.data_parallel,
            mock_embeddings=args.mock_embeddings,
            mock_dim=args.mock_dim,
            alpha=0.6,
            hops=2,
            top_k=args.top_k,
        )

        for _, row in df.iterrows():
            entry = row.to_dict()
            entry["scenario"] = scenario["name"]
            summary_rows.append(entry)

    import pandas as pd

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "synthetic_suite_summary.csv", index=False)
    print(f"Saved {output_root / 'synthetic_suite_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
