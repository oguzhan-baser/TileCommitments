#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

from protocol import get_json, load_fixed_models, post_json, verify_prover_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verifier web UI. Provides a dropdown model selector, sends query to prover server, "
            "then verifies returned proof bundles locally."
        )
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--prover-url", type=str, default="http://127.0.0.1:8081")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=Path(__file__).resolve().parent / "fixed_models.json",
        help="Fallback model list if prover /models is unavailable.",
    )
    parser.add_argument("--default-prompt", type=str, default="Where is the capital of the world?")
    parser.add_argument("--default-max-new-tokens", type=int, default=16)
    parser.add_argument("--default-num-queries", type=int, default=10)
    return parser.parse_args()


class VerifierWebServer(ThreadingHTTPServer):
    def __init__(self, server_address, request_handler_class, config: Dict[str, Any]):
        super().__init__(server_address, request_handler_class)
        self.config = config


def _coerce_int(values: Dict[str, List[str]], key: str, default: int) -> int:
    raw = values.get(key, [str(default)])[0]
    try:
        return int(raw)
    except ValueError:
        return default


def _escape(text: Any) -> str:
    return html.escape(str(text))


def render_page(
    *,
    models: List[str],
    selected_model: str,
    prompt: str,
    max_new_tokens: int,
    num_queries: int,
    prover_url: str,
    message: str = "",
    verification_payload: Dict[str, Any] | None = None,
) -> str:
    options_html = "\n".join(
        f"<option value='{_escape(m)}' {'selected' if m == selected_model else ''}>{_escape(m)}</option>"
        for m in models
    )
    message_html = f"<div style='padding:10px;background:#eef;border-radius:8px;margin-bottom:12px'>{_escape(message)}</div>" if message else ""

    results_html = ""
    if verification_payload is not None:
        verification = verification_payload["verification"]
        rows = []
        for row in verification["proof_results"]:
            status = "✅" if row["verified"] else "❌"
            rows.append(
                "<tr>"
                f"<td>{row['bundle_id']}</td>"
                f"<td>{_escape(row['index'])}</td>"
                f"<td>{_escape(row['value_int'])}</td>"
                f"<td>{row['proof_elements']}</td>"
                f"<td>{status}</td>"
                "</tr>"
            )
        rows_html = "\n".join(rows)
        generated_text = verification_payload.get("inference_output", {}).get("generated_text", "")
        summary_json = _escape(json.dumps(verification_payload, indent=2))
        results_html = f"""
        <h3>Verification Result</h3>
        <p><b>Model:</b> {_escape(verification_payload.get("model_name", ""))}</p>
        <p><b>All proofs verified:</b> {'✅' if verification['all_verified'] else '❌'}</p>
        <p><b>Proof count:</b> {verification['num_proofs']}</p>
        <p><b>Inference output:</b> {_escape(generated_text)}</p>
        <table border="1" cellpadding="6" cellspacing="0">
          <thead>
            <tr><th>Bundle</th><th>Index</th><th>Value</th><th>Proof elems</th><th>Verified</th></tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        <details style="margin-top:12px">
          <summary>Full verification JSON</summary>
          <pre>{summary_json}</pre>
        </details>
        """

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SplitCompute Verifier UI</title>
  </head>
  <body style="font-family:Arial,sans-serif;max-width:980px;margin:30px auto;line-height:1.4">
    <h2>SplitCompute Verifier (Web Interface)</h2>
    <p><b>Prover URL:</b> {_escape(prover_url)}</p>
    {message_html}
    <form method="POST" action="/verify">
      <label>Model</label><br/>
      <select name="model_name" style="min-width:420px">{options_html}</select><br/><br/>
      <label>Prompt</label><br/>
      <textarea name="prompt" rows="4" cols="110">{_escape(prompt)}</textarea><br/><br/>
      <label>Max new tokens</label><br/>
      <input type="number" name="max_new_tokens" value="{max_new_tokens}" min="1" /><br/><br/>
      <label>Num commitment proofs</label><br/>
      <input type="number" name="num_queries" value="{num_queries}" min="1" /><br/><br/>
      <button type="submit">Query Prover and Verify</button>
    </form>
    <hr/>
    {results_html}
  </body>
</html>
"""


class VerifierUIHandler(BaseHTTPRequestHandler):
    server: VerifierWebServer

    def _send_html(self, status: int, html_body: str) -> None:
        body = html_body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_models(self) -> List[str]:
        prover_url = self.server.config["prover_url"].rstrip("/")
        try:
            models_payload = get_json(prover_url + "/models", timeout_s=10)
            models = models_payload.get("models", [])
            if isinstance(models, list) and models:
                return [str(v) for v in models]
        except Exception:  # noqa: BLE001
            pass
        return list(self.server.config["fallback_models"])

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/", "/index.html"}:
            self._send_html(HTTPStatus.NOT_FOUND, "<h3>Not found</h3>")
            return
        models = self._resolve_models()
        selected_model = models[0]
        html_body = render_page(
            models=models,
            selected_model=selected_model,
            prompt=self.server.config["default_prompt"],
            max_new_tokens=self.server.config["default_max_new_tokens"],
            num_queries=self.server.config["default_num_queries"],
            prover_url=self.server.config["prover_url"],
        )
        self._send_html(HTTPStatus.OK, html_body)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/verify":
            self._send_html(HTTPStatus.NOT_FOUND, "<h3>Not found</h3>")
            return

        models = self._resolve_models()
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b""
            form = parse_qs(raw.decode("utf-8"))

            selected_model = form.get("model_name", [models[0]])[0]
            prompt = form.get("prompt", [self.server.config["default_prompt"]])[0].strip()
            max_new_tokens = _coerce_int(form, "max_new_tokens", self.server.config["default_max_new_tokens"])
            num_queries = _coerce_int(form, "num_queries", self.server.config["default_num_queries"])

            request_payload = {
                "model_name": selected_model,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "num_queries": num_queries,
                "skip_interp_build": True,
            }
            prover_url = self.server.config["prover_url"].rstrip("/")
            prover_response = post_json(prover_url + "/prove", request_payload, timeout_s=7200)
            verification_report = verify_prover_response(prover_response)
            html_body = render_page(
                models=models,
                selected_model=selected_model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_queries=num_queries,
                prover_url=self.server.config["prover_url"],
                message="Proof bundles received and verified.",
                verification_payload=verification_report,
            )
            self._send_html(HTTPStatus.OK, html_body)
        except Exception as exc:  # noqa: BLE001
            html_body = render_page(
                models=models,
                selected_model=models[0],
                prompt=self.server.config["default_prompt"],
                max_new_tokens=self.server.config["default_max_new_tokens"],
                num_queries=self.server.config["default_num_queries"],
                prover_url=self.server.config["prover_url"],
                message=f"Error: {exc}",
            )
            self._send_html(HTTPStatus.BAD_REQUEST, html_body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    args = parse_args()
    fallback_models = load_fixed_models(args.models_file)
    config = {
        "prover_url": args.prover_url,
        "fallback_models": fallback_models,
        "default_prompt": args.default_prompt,
        "default_max_new_tokens": int(args.default_max_new_tokens),
        "default_num_queries": int(args.default_num_queries),
    }
    server = VerifierWebServer((args.host, args.port), VerifierUIHandler, config=config)
    print(f"[INFO] Verifier web UI listening on http://{args.host}:{args.port}")
    print(f"[INFO] Prover URL: {args.prover_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()

