#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PIPELINE_SCRIPT = SCRIPT_DIR / "multiprover_pipeline.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prover-count sweep for the multiprover pipeline and collect aggregate experiment metrics."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--min-provers", type=int, default=1, help="Starting prover count")
    parser.add_argument("--max-provers", type=int, default=10, help="Ending prover count")
    parser.add_argument(
        "--device-ids",
        type=str,
        default=None,
        help="Comma-separated CUDA ids to form the prover pool (default: all visible GPUs).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Computation dtype",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for commitment sampling")
    parser.add_argument("--scale-factor", type=int, default=16, help="Scale factor for float->int conversion")
    parser.add_argument("--quantize", type=float, default=50.0, help="Quantization percentage")
    parser.add_argument("--min-dim", type=int, default=4, help="Hypercube min dimension")
    parser.add_argument("--max-dim", type=int, default=10, help="Hypercube max dimension")
    parser.add_argument("--num-queries", type=int, default=10, help="Sampled prove/verify queries")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Allclose rtol")
    parser.add_argument("--atol", type=float, default=5e-2, help="Allclose atol")
    parser.add_argument("--skip-interp-build", action="store_true", help="Pass --skip-build to interpolation stage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Sweep output directory (default: multiproversetting/output/sweeps/sweep_YYYYmmdd_HHMMSS_utc)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop immediately when one sweep point fails (default: continue and record status).",
    )
    return parser.parse_args()


def resolve_output_dir(path_arg: Path | None) -> Path:
    if path_arg is not None:
        return path_arg.resolve()
    tag = datetime.now(timezone.utc).strftime("sweep_%Y%m%d_%H%M%S_utc")
    return (REPO_ROOT / "multiproversetting" / "output" / "sweeps" / tag).resolve()


def parse_device_pool(device_ids_arg: str | None) -> List[int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multiprover sweep, but no GPU is visible.")
    total = torch.cuda.device_count()
    if device_ids_arg is None:
        return list(range(total))

    pool: List[int] = []
    for raw in device_ids_arg.split(","):
        raw = raw.strip()
        if not raw:
            continue
        idx = int(raw)
        if idx < 0 or idx >= total:
            raise ValueError(f"Invalid CUDA index {idx}; available range is [0, {total - 1}]")
        if idx in pool:
            raise ValueError(f"Duplicate device id detected in --device-ids: {idx}")
        pool.append(idx)
    if not pool:
        raise ValueError("--device-ids resolved to an empty pool")
    return pool


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean_from(items: List[Dict[str, Any]], key: str) -> float:
    vals = [_safe_float(item.get("timing", {}).get(key, 0.0)) for item in items]
    return float(mean(vals)) if vals else 0.0


def summarize_run(overhead: Dict[str, Any], num_provers: int, run_dir: Path) -> Dict[str, Any]:
    stage = overhead.get("stage_timings_seconds", {})
    multi_compute_summary = overhead.get("multi_prover_compute_summary", {})
    exchange = overhead.get("embedding_exchange_summary", {})
    multi_commit = overhead.get("multi_prover_commit_summary", [])

    avg_agent_compute = multi_compute_summary.get("avg_inference_time_across_agents_seconds")
    if avg_agent_compute is None:
        per_agent = multi_compute_summary.get("per_prover_compute_seconds", [])
        if per_agent:
            avg_agent_compute = float(mean([_safe_float(v) for v in per_agent]))
        else:
            total_multi_compute = _safe_float(stage.get("multi_prover_compute", 0.0))
            avg_agent_compute = float(total_multi_compute / max(num_provers, 1))

    summary = {
        "status": "success",
        "num_provers": int(num_provers),
        "run_dir": str(run_dir),
        "overhead_report_json": str(run_dir / "overhead_report.json"),
        "one_prover_compute_s": _safe_float(stage.get("one_prover_compute", 0.0)),
        "multi_prover_compute_s": _safe_float(stage.get("multi_prover_compute", 0.0)),
        "one_prover_commit_pipeline_s": _safe_float(stage.get("one_prover_commit_pipeline_total", 0.0)),
        "multi_prover_commit_pipeline_s": _safe_float(stage.get("multi_prover_commit_pipeline_total", 0.0)),
        "one_prover_total_s": _safe_float(stage.get("one_prover_total_compute_plus_commit", 0.0)),
        "multi_prover_total_s": _safe_float(stage.get("multi_prover_total_compute_plus_commit", 0.0)),
        "avg_agent_inference_s": _safe_float(avg_agent_compute),
        "avg_agent_commit_s": _mean_from(multi_commit, "commit_time_s"),
        "avg_agent_total_prove_s": _mean_from(multi_commit, "total_prove_time_s"),
        "avg_agent_total_verify_s": _mean_from(multi_commit, "total_verify_time_s"),
        "avg_exchange_bytes": _safe_float(exchange.get("average_exchange_bytes", 0.0)),
        "avg_exchange_mib": _safe_float(exchange.get("average_exchange_mib", 0.0)),
        "total_exchange_bytes": int(exchange.get("total_exchange_bytes", 0) or 0),
        "total_exchange_mib": _safe_float(exchange.get("total_exchange_mib", 0.0)),
        "num_exchanges": int(exchange.get("num_exchanges", 0) or 0),
        "one_vs_multi_next_token_match": bool(overhead.get("one_vs_multi_match", {}).get("next_token_match", False)),
    }
    return summary


def run_single_experiment(args: argparse.Namespace, sweep_dir: Path, num_provers: int, device_pool: List[int]) -> Dict[str, Any]:
    run_dir = sweep_dir / f"provers_{num_provers:02d}"
    if num_provers > len(device_pool):
        return {
            "status": "skipped",
            "num_provers": int(num_provers),
            "reason": (
                f"Requested {num_provers} distinct provers, but only {len(device_pool)} distinct GPU(s) "
                "available in the selected device pool."
            ),
            "run_dir": str(run_dir),
        }

    device_ids = ",".join(str(idx) for idx in device_pool[:num_provers])
    cmd = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--num-provers",
        str(num_provers),
        "--device-ids",
        device_ids,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--scale-factor",
        str(args.scale_factor),
        "--quantize",
        str(args.quantize),
        "--min-dim",
        str(args.min_dim),
        "--max-dim",
        str(args.max_dim),
        "--num-queries",
        str(args.num_queries),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--output-dir",
        str(run_dir),
    ]
    if args.skip_interp_build:
        cmd.append("--skip-interp-build")

    print(f"[sweep] Running num_provers={num_provers} on devices={device_ids}")
    start = time.perf_counter()
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        return {
            "status": "failed",
            "num_provers": int(num_provers),
            "reason": str(exc),
            "run_dir": str(run_dir),
            "wall_seconds": float(time.perf_counter() - start),
        }

    wall_seconds = float(time.perf_counter() - start)
    overhead_path = run_dir / "overhead_report.json"
    if not overhead_path.is_file():
        return {
            "status": "failed",
            "num_provers": int(num_provers),
            "reason": f"Missing expected artifact: {overhead_path}",
            "run_dir": str(run_dir),
            "wall_seconds": wall_seconds,
        }

    with overhead_path.open("r", encoding="utf-8") as handle:
        overhead = json.load(handle)
    row = summarize_run(overhead, num_provers=num_provers, run_dir=run_dir)
    row["wall_seconds"] = wall_seconds
    return row


def save_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    field_order = [
        "status",
        "num_provers",
        "run_dir",
        "overhead_report_json",
        "wall_seconds",
        "one_prover_compute_s",
        "multi_prover_compute_s",
        "one_prover_commit_pipeline_s",
        "multi_prover_commit_pipeline_s",
        "one_prover_total_s",
        "multi_prover_total_s",
        "avg_agent_inference_s",
        "avg_agent_commit_s",
        "avg_agent_total_prove_s",
        "avg_agent_total_verify_s",
        "avg_exchange_bytes",
        "avg_exchange_mib",
        "total_exchange_bytes",
        "total_exchange_mib",
        "num_exchanges",
        "one_vs_multi_next_token_match",
        "reason",
    ]

    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = [k for k in field_order if k in keys] + [k for k in sorted(keys) if k not in field_order]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.min_provers < 1 or args.max_provers < args.min_provers:
        raise ValueError("Require 1 <= min-provers <= max-provers")

    sweep_dir = resolve_output_dir(args.output_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    device_pool = parse_device_pool(args.device_ids)
    print(f"[sweep] Device pool: {device_pool}")
    print(f"[sweep] Running prover counts from {args.min_provers} to {args.max_provers}")

    rows: List[Dict[str, Any]] = []
    for num_provers in range(args.min_provers, args.max_provers + 1):
        row = run_single_experiment(args, sweep_dir, num_provers, device_pool)
        rows.append(row)
        print(f"[sweep] num_provers={num_provers} -> {row['status']}")
        if row["status"] == "failed" and args.stop_on_failure:
            break

    successful = [row for row in rows if row.get("status") == "success"]
    skipped = [row for row in rows if row.get("status") == "skipped"]
    failed = [row for row in rows if row.get("status") == "failed"]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": args.model,
            "prompt": args.prompt,
            "min_provers": args.min_provers,
            "max_provers": args.max_provers,
            "device_pool": device_pool,
            "dtype": args.dtype,
            "seed": args.seed,
            "scale_factor": args.scale_factor,
            "quantize": args.quantize,
            "min_dim": args.min_dim,
            "max_dim": args.max_dim,
            "num_queries": args.num_queries,
            "rtol": args.rtol,
            "atol": args.atol,
            "skip_interp_build": bool(args.skip_interp_build),
            "stop_on_failure": bool(args.stop_on_failure),
        },
        "counts": {
            "success": len(successful),
            "skipped": len(skipped),
            "failed": len(failed),
            "total": len(rows),
        },
        "runs": rows,
    }

    summary_json = sweep_dir / "sweep_results.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    csv_path = sweep_dir / "sweep_results.csv"
    save_csv(rows, csv_path)

    print(f"[sweep] Saved summary: {summary_json}")
    print(f"[sweep] Saved CSV: {csv_path}")
    print(
        f"[sweep] Counts -> success={len(successful)}, skipped={len(skipped)}, failed={len(failed)}, total={len(rows)}"
    )

    if not successful:
        raise RuntimeError("Sweep completed but produced no successful runs.")


if __name__ == "__main__":
    main()

