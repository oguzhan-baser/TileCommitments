#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot multiprover sweep results as a three-panel figure: "
            "agent inference time, agent commit/prove/verify times, and exchanged embedding size."
        )
    )
    parser.add_argument(
        "--sweep-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more sweep_results.json files from sweep_multiprover_experiments.py",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Directory for generated plots and merged CSV",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="Optional figure title prefix",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model = payload.get("config", {}).get("model", "unknown_model")
    rows = payload.get("runs", [])
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "success":
            continue
        out.append(
            {
                "model": model,
                "num_provers": int(row["num_provers"]),
                "avg_agent_inference_s": float(row.get("avg_agent_inference_s", 0.0)),
                "avg_agent_commit_s": float(row.get("avg_agent_commit_s", 0.0)),
                "avg_agent_total_prove_s": float(row.get("avg_agent_total_prove_s", 0.0)),
                "avg_agent_total_verify_s": float(row.get("avg_agent_total_verify_s", 0.0)),
                "avg_exchange_bytes": float(row.get("avg_exchange_bytes", 0.0)),
                "avg_exchange_mib": float(row.get("avg_exchange_mib", 0.0)),
                "run_dir": str(row.get("run_dir", "")),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, Any]] = []
    for sweep_path in args.sweep_json:
        path = sweep_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing sweep results file: {path}")
        merged_rows.extend(load_rows(path))

    if not merged_rows:
        raise RuntimeError("No successful runs found in provided sweep_results.json files.")

    df = pd.DataFrame(merged_rows).sort_values(["model", "num_provers"]).reset_index(drop=True)
    merged_csv = outdir / "multiprover_sweep_merged.csv"
    df.to_csv(merged_csv, index=False)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_agent_inference_s",
        hue="model",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("A) Avg Agent Inference Time")
    axes[0].set_xlabel("Number of provers")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(sorted(df["num_provers"].unique()))

    timing_map = {
        "avg_agent_commit_s": "commit",
        "avg_agent_total_prove_s": "prove",
        "avg_agent_total_verify_s": "verify",
    }
    timing_long = df.melt(
        id_vars=["model", "num_provers"],
        value_vars=list(timing_map.keys()),
        var_name="timing_metric",
        value_name="seconds",
    )
    timing_long["timing_metric"] = timing_long["timing_metric"].map(timing_map)
    timing_long["series"] = timing_long["model"] + " | " + timing_long["timing_metric"]

    sns.lineplot(
        data=timing_long,
        x="num_provers",
        y="seconds",
        hue="series",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("B) Avg Agent Commit/Prove/Verify")
    axes[1].set_xlabel("Number of provers")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticks(sorted(df["num_provers"].unique()))

    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_exchange_mib",
        hue="model",
        marker="o",
        ax=axes[2],
    )
    axes[2].set_title("C) Avg Embedding Exchange Size")
    axes[2].set_xlabel("Number of provers")
    axes[2].set_ylabel("MiB per transfer")
    axes[2].set_xticks(sorted(df["num_provers"].unique()))

    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()

    fig_path = outdir / "multiprover_sweep_three_panel.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] Saved merged CSV: {merged_csv}")
    print(f"[plot] Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

