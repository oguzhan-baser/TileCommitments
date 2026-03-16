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
            "Plot multiprover sweep results as 5 separate figures: "
            "agent inference time, agent commit time, agent prove time, "
            "agent verify time, and exchanged embedding size."
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
    x_ticks = sorted(df["num_provers"].unique())

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_agent_inference_s",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("A) Avg Agent Inference Time")
    ax.set_xlabel("Number of provers")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x_ticks)
    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    fig_path_infer = outdir / "multiprover_sweep_a_avg_agent_inference_time.png"
    fig.savefig(fig_path_infer, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_agent_commit_s",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("B) Avg Agent Commit Time")
    ax.set_xlabel("Number of provers")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x_ticks)
    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    fig_path_commit = outdir / "multiprover_sweep_b_avg_agent_commit_time.png"
    fig.savefig(fig_path_commit, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_agent_total_prove_s",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("C) Avg Agent Prove Time")
    ax.set_xlabel("Number of provers")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x_ticks)
    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    fig_path_prove = outdir / "multiprover_sweep_c_avg_agent_prove_time.png"
    fig.savefig(fig_path_prove, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_agent_total_verify_s",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("D) Avg Agent Verify Time")
    ax.set_xlabel("Number of provers")
    ax.set_ylabel("Seconds")
    ax.set_xticks(x_ticks)
    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    fig_path_verify = outdir / "multiprover_sweep_d_avg_agent_verify_time.png"
    fig.savefig(fig_path_verify, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df,
        x="num_provers",
        y="avg_exchange_mib",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("E) Avg Embedding Exchange Size")
    ax.set_xlabel("Number of provers")
    ax.set_ylabel("MiB per transfer")
    ax.set_xticks(x_ticks)
    if args.title_prefix:
        fig.suptitle(args.title_prefix, y=1.02)
    fig.tight_layout()
    fig_path_exchange = outdir / "multiprover_sweep_e_avg_embedding_exchange_size.png"
    fig.savefig(fig_path_exchange, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot] Saved merged CSV: {merged_csv}")
    print(f"[plot] Saved figure: {fig_path_infer}")
    print(f"[plot] Saved figure: {fig_path_commit}")
    print(f"[plot] Saved figure: {fig_path_prove}")
    print(f"[plot] Saved figure: {fig_path_verify}")
    print(f"[plot] Saved figure: {fig_path_exchange}")


if __name__ == "__main__":
    main()
