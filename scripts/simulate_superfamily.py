#!/usr/bin/env python3
"""Generate synthetic B1 superfamily datasets with planted hotspot signals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")
POLAR = set("STNQYC")
CHARGED = set("KRHDE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a synthetic B1 superfamily dataset")
    parser.add_argument("--output-dir", default="output/synthetic/example", help="Output directory")
    parser.add_argument("--families", default="FAM1,FAM2,FAM3", help="Comma-separated family names")
    parser.add_argument("--seqs-per-family", type=int, default=20, help="Sequences per family (incl. reference)")
    parser.add_argument("--length", type=int, default=250, help="Sequence length")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--composition", choices=["neutral", "hydrophobic", "polar", "charged"], default="neutral")
    parser.add_argument("--motif", default="HXD", help="Motif to plant as hotspot")
    parser.add_argument("--motif-start", type=int, default=None, help="1-indexed start position for motif")
    parser.add_argument(
        "--motif-positions",
        default=None,
        help="Comma-separated 1-indexed positions for each motif character",
    )
    parser.add_argument("--background-rate", type=float, default=0.01, help="Background mutation rate")
    parser.add_argument("--hotspot-rate", type=float, default=0.25, help="Hotspot mutation rate")
    return parser.parse_args()


def aa_weights(composition: str) -> List[float]:
    weights = []
    for aa in AMINO_ACIDS:
        if composition == "hydrophobic" and aa in HYDROPHOBIC:
            weights.append(3.0)
        elif composition == "polar" and aa in POLAR:
            weights.append(3.0)
        elif composition == "charged" and aa in CHARGED:
            weights.append(3.0)
        else:
            weights.append(1.0)
    total = sum(weights)
    return [w / total for w in weights]


def generate_sequence(length: int, rng: np.random.Generator, composition: str) -> str:
    weights = aa_weights(composition)
    return "".join(rng.choice(AMINO_ACIDS, size=length, p=weights))


def plant_motif(sequence: str, motif: str, positions: Optional[List[int]], rng: np.random.Generator) -> Tuple[str, List[int]]:
    seq = list(sequence)
    length = len(seq)
    motif_len = len(motif)

    if positions:
        if len(positions) != motif_len:
            raise ValueError("motif-positions must match motif length")
        pos0 = [p - 1 for p in positions]
    else:
        start = rng.integers(0, max(1, length - motif_len + 1))
        pos0 = list(range(start, start + motif_len))

    for idx, aa in zip(pos0, motif):
        if 0 <= idx < length:
            seq[idx] = aa

    return "".join(seq), pos0


def mutate_sequence(
    reference: str,
    hotspot_positions: List[int],
    rng: np.random.Generator,
    background_rate: float,
    hotspot_rate: float,
) -> str:
    seq = list(reference)
    for idx, aa in enumerate(reference):
        rate = hotspot_rate if idx in hotspot_positions else background_rate
        if rng.random() < rate:
            choices = [x for x in AMINO_ACIDS if x != aa]
            seq[idx] = rng.choice(choices)
    return "".join(seq)


def simulate_dataset(
    output_dir: Path,
    families: List[str],
    seqs_per_family: int,
    length: int,
    seed: int,
    composition: str,
    motif: str,
    motif_positions: Optional[List[int]],
    background_rate: float,
    hotspot_rate: float,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_lines: List[str] = []
    manifest_rows: List[Dict[str, object]] = []
    labels: Dict[str, object] = {
        "seed": seed,
        "length": length,
        "composition": composition,
        "motif": motif,
        "background_rate": background_rate,
        "hotspot_rate": hotspot_rate,
        "families": {},
    }

    if seqs_per_family < 2:
        raise ValueError("seqs-per-family must be at least 2")

    for fam in families:
        base_seq = generate_sequence(length, rng, composition)
        base_seq, hotspots = plant_motif(base_seq, motif, motif_positions, rng)
        labels["families"][fam] = {
            "hotspot_positions_0idx": hotspots,
            "hotspot_positions_1idx": [p + 1 for p in hotspots],
        }

        ref_id = f"{fam}|REF"
        fasta_lines.append(f">{ref_id}\n{base_seq}")
        manifest_rows.append(
            {
                "header": ref_id,
                "family": fam,
                "length": length,
                "is_reference": 1,
                "hotspot_count": len(hotspots),
            }
        )

        for i in range(seqs_per_family - 1):
            var_id = f"{fam}|VAR{i+1:04d}"
            var_seq = mutate_sequence(base_seq, hotspots, rng, background_rate, hotspot_rate)
            fasta_lines.append(f">{var_id}\n{var_seq}")
            manifest_rows.append(
                {
                    "header": var_id,
                    "family": fam,
                    "length": length,
                    "is_reference": 0,
                    "hotspot_count": len(hotspots),
                }
            )

    fasta_path = output_dir / "synthetic.fasta"
    fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    import pandas as pd

    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)

    return {
        "fasta": str(fasta_path),
        "labels": str(output_dir / "labels.json"),
        "manifest": str(output_dir / "manifest.csv"),
        "families": families,
    }


def main() -> int:
    args = parse_args()
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    motif_positions = None
    if args.motif_positions:
        motif_positions = [int(p) for p in args.motif_positions.split(",") if p.strip()]

    simulate_dataset(
        output_dir=Path(args.output_dir),
        families=families,
        seqs_per_family=args.seqs_per_family,
        length=args.length,
        seed=args.seed,
        composition=args.composition,
        motif=args.motif,
        motif_positions=motif_positions,
        background_rate=args.background_rate,
        hotspot_rate=args.hotspot_rate,
    )
    print(f"Saved synthetic dataset to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
