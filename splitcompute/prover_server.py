#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from protocol import load_fixed_models, run_prover_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prover server: receives model+prompt, runs activation->commitment pipeline, "
            "returns inference output + commitment + proof bundles."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8081, help="Bind port")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=Path(__file__).resolve().parent / "fixed_models.json",
        help="JSON file with fixed model allowlist.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "prover_runs",
        help="Where prover run artifacts are saved.",
    )
    parser.add_argument("--default-max-new-tokens", type=int, default=16)
    parser.add_argument("--default-num-queries", type=int, default=10)
    parser.add_argument("--default-seed", type=int, default=42)
    parser.add_argument("--default-scale-factor", type=int, default=16)
    parser.add_argument("--default-quantize", type=float, default=50.0)
    parser.add_argument("--default-min-dim", type=int, default=4)
    parser.add_argument("--default-max-dim", type=int, default=10)
    parser.add_argument(
        "--skip-interp-build",
        action="store_true",
        default=True,
        help="Pass --skip-build to interpolation stage (default enabled).",
    )
    return parser.parse_args()


class ProverHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, config: Dict[str, Any]):
        super().__init__(server_address, request_handler_class)
        self.config = config


class ProverHandler(BaseHTTPRequestHandler):
    server: ProverHTTPServer

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/models":
            self._send_json(
                HTTPStatus.OK,
                {"models": self.server.config["allowed_models"]},
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/prove":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown path: {parsed.path}"})
            return

        try:
            payload = self._read_json_body()
            model_name = str(payload.get("model_name", "")).strip()
            prompt = str(payload.get("prompt", "")).strip()
            if not model_name:
                raise ValueError("Missing required field: model_name")
            if not prompt:
                raise ValueError("Missing required field: prompt")

            response_payload = run_prover_pipeline(
                model_name=model_name,
                prompt=prompt,
                output_root=Path(self.server.config["output_root"]),
                allowed_models=list(self.server.config["allowed_models"]),
                max_new_tokens=int(payload.get("max_new_tokens", self.server.config["default_max_new_tokens"])),
                num_queries=int(payload.get("num_queries", self.server.config["default_num_queries"])),
                seed=int(payload.get("seed", self.server.config["default_seed"])),
                scale_factor=int(payload.get("scale_factor", self.server.config["default_scale_factor"])),
                quantize=float(payload.get("quantize", self.server.config["default_quantize"])),
                min_dim=int(payload.get("min_dim", self.server.config["default_min_dim"])),
                max_dim=int(payload.get("max_dim", self.server.config["default_max_dim"])),
                skip_interp_build=bool(payload.get("skip_interp_build", self.server.config["skip_interp_build"])),
            )
            self._send_json(HTTPStatus.OK, response_payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "error": str(exc),
                },
            )

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    args = parse_args()
    allowed_models = load_fixed_models(args.models_file)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "allowed_models": allowed_models,
        "output_root": str(output_root),
        "default_max_new_tokens": int(args.default_max_new_tokens),
        "default_num_queries": int(args.default_num_queries),
        "default_seed": int(args.default_seed),
        "default_scale_factor": int(args.default_scale_factor),
        "default_quantize": float(args.default_quantize),
        "default_min_dim": int(args.default_min_dim),
        "default_max_dim": int(args.default_max_dim),
        "skip_interp_build": bool(args.skip_interp_build),
    }

    server = ProverHTTPServer((args.host, args.port), ProverHandler, config=config)
    print(f"[INFO] Prover server listening on http://{args.host}:{args.port}")
    print(f"[INFO] Allowed models: {allowed_models}")
    print(f"[INFO] Output root: {output_root}")
    server.serve_forever()


if __name__ == "__main__":
    main()

