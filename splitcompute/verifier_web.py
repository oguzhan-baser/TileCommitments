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
    message_kind = "error" if str(message).lower().startswith("error") else "success"
    message_html = (
        f"<div class='status {message_kind}'>{_escape(message)}</div>"
        if message
        else ""
    )

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
        <section class="card">
          <h3>Verification Result</h3>
          <div class="kv-grid">
            <div><span class="k">Model</span><span class="v">{_escape(verification_payload.get("model_name", ""))}</span></div>
            <div><span class="k">All proofs verified</span><span class="v">{'✅' if verification['all_verified'] else '❌'}</span></div>
            <div><span class="k">Proof count</span><span class="v">{verification['num_proofs']}</span></div>
            <div><span class="k">Inference output</span><span class="v">{_escape(generated_text)}</span></div>
          </div>
          <div class="table-wrap">
            <table>
              <thead>
                <tr><th>Bundle</th><th>Index</th><th>Value</th><th>Proof elems</th><th>Verified</th></tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
          </div>
          <details>
            <summary>Full verification JSON</summary>
            <pre>{summary_json}</pre>
          </details>
        </section>
        """

    return f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>SplitCompute Verifier UI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      :root {{
        --bg-1: #060a14;
        --bg-2: #0a1226;
        --panel: rgba(18, 26, 44, 0.78);
        --panel-border: rgba(119, 224, 255, 0.25);
        --text: #e8efff;
        --muted: #9fb3d9;
        --accent: #7af4ff;
        --accent-2: #6bffbc;
        --danger: #ff6e8d;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        color: var(--text);
        background:
          radial-gradient(1200px 600px at 0% 0%, rgba(85, 183, 255, 0.22), transparent 60%),
          radial-gradient(900px 500px at 100% 0%, rgba(120, 255, 197, 0.18), transparent 55%),
          linear-gradient(180deg, var(--bg-2), var(--bg-1));
        min-height: 100vh;
      }}
      .container {{
        max-width: 1080px;
        margin: 28px auto 42px;
        padding: 0 16px;
      }}
      .hero {{
        margin-bottom: 16px;
      }}
      .eyebrow {{
        font-size: 12px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--accent-2);
        margin-bottom: 8px;
      }}
      h1 {{
        margin: 0 0 6px;
        font-size: clamp(26px, 3.2vw, 38px);
        line-height: 1.1;
      }}
      .subtitle {{
        margin: 0;
        color: var(--muted);
      }}
      .card {{
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 16px;
        backdrop-filter: blur(10px);
        padding: 16px;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.35);
        margin-bottom: 14px;
      }}
      .status {{
        padding: 10px 12px;
        border-radius: 10px;
        margin-bottom: 12px;
        font-size: 14px;
      }}
      .status.success {{
        background: rgba(77, 246, 179, 0.14);
        border: 1px solid rgba(77, 246, 179, 0.35);
      }}
      .status.error {{
        background: rgba(255, 110, 141, 0.14);
        border: 1px solid rgba(255, 110, 141, 0.45);
      }}
      label {{
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin: 10px 0 6px;
      }}
      select, textarea, input {{
        width: 100%;
        background: rgba(10, 17, 31, 0.9);
        color: var(--text);
        border: 1px solid rgba(131, 170, 255, 0.28);
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 14px;
        outline: none;
      }}
      select:focus, textarea:focus, input:focus {{
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(122, 244, 255, 0.18);
      }}
      .grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
      }}
      .btn {{
        margin-top: 14px;
        border: 0;
        border-radius: 10px;
        padding: 11px 16px;
        background: linear-gradient(90deg, #74f2ff, #78ffcb);
        color: #051019;
        font-weight: 700;
        cursor: pointer;
      }}
      .btn:disabled {{
        opacity: 0.75;
        cursor: wait;
      }}
      .prover-url {{
        margin-top: 12px;
        font-size: 13px;
        color: var(--muted);
      }}
      .kv-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 10px;
        margin-bottom: 12px;
      }}
      .kv-grid > div {{
        border: 1px solid rgba(137, 170, 255, 0.22);
        border-radius: 10px;
        padding: 10px;
        background: rgba(13, 21, 37, 0.8);
      }}
      .k {{
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 4px;
      }}
      .v {{
        display: block;
        font-size: 14px;
      }}
      .table-wrap {{
        overflow-x: auto;
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8px;
      }}
      th, td {{
        border-bottom: 1px solid rgba(138, 170, 255, 0.2);
        padding: 8px 10px;
        text-align: left;
        font-size: 13px;
      }}
      th {{
        color: var(--accent);
      }}
      details {{
        margin-top: 12px;
      }}
      pre {{
        white-space: pre-wrap;
        word-break: break-word;
        background: rgba(9, 16, 27, 0.92);
        border: 1px solid rgba(133, 160, 255, 0.22);
        border-radius: 10px;
        padding: 10px;
        max-height: 300px;
        overflow: auto;
      }}
      .loading-overlay {{
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(4, 8, 16, 0.82);
        backdrop-filter: blur(3px);
      }}
      .loading-overlay.hidden {{
        display: none;
      }}
      .loading-card {{
        width: min(520px, 92vw);
        background: rgba(12, 19, 33, 0.96);
        border: 1px solid rgba(122, 244, 255, 0.3);
        border-radius: 16px;
        padding: 18px 16px;
        text-align: center;
      }}
      .spinner {{
        width: 54px;
        height: 54px;
        border-radius: 50%;
        border: 3px solid rgba(122, 244, 255, 0.28);
        border-top-color: #76f7ff;
        margin: 6px auto 12px;
        animation: spin 0.85s linear infinite;
      }}
      @keyframes spin {{
        to {{ transform: rotate(360deg); }}
      }}
      .loading-title {{
        margin: 0 0 6px;
        font-size: 18px;
      }}
      .loading-subtitle {{
        margin: 0;
        color: var(--muted);
        font-size: 13px;
      }}
      @media (max-width: 760px) {{
        .grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="loading-overlay hidden" id="loading-overlay">
      <div class="loading-card">
        <div class="spinner"></div>
        <h3 class="loading-title">Verifier is waiting for prover results</h3>
        <p class="loading-subtitle" id="loading-subtitle">
          Running inference, generating commitment, and validating proofs...
        </p>
      </div>
    </div>
    <main class="container">
      <section class="hero">
        <div class="eyebrow">Multivariate Commitment/Verification Protocol</div>
        <h1>Verifier Portal</h1>
        <p class="subtitle">TensorCommitments for Theseus Agents.</p>
      </section>
      <section class="card">
        {message_html}
        <form method="POST" action="/verify" id="verify-form">
          <label>Model</label>
          <select name="model_name">{options_html}</select>
          <label>Prompt</label>
          <textarea name="prompt" rows="5">{_escape(prompt)}</textarea>
          <div class="grid">
            <div>
              <label>Max new tokens</label>
              <input type="number" name="max_new_tokens" value="{max_new_tokens}" min="1" />
            </div>
            <div>
              <label>Num commitment proofs</label>
              <input type="number" name="num_queries" value="{num_queries}" min="1" />
            </div>
          </div>
          <button type="submit" class="btn" id="submit-btn">Query Prover and Verify</button>
          <p class="prover-url"><b>Prover URL:</b> {_escape(prover_url)}</p>
        </form>
      </section>
      {results_html}
    </main>
    <script>
      (function() {{
        const form = document.getElementById('verify-form');
        const overlay = document.getElementById('loading-overlay');
        const submitBtn = document.getElementById('submit-btn');
        const subtitle = document.getElementById('loading-subtitle');
        const frames = [
          'Running inference, generating commitment, and validating proofs',
          'Still working — proving and verification may take a while for larger models',
          'Finalizing cryptographic checks and preparing response'
        ];
        let ticker = null;
        form.addEventListener('submit', function() {{
          overlay.classList.remove('hidden');
          submitBtn.disabled = true;
          submitBtn.textContent = 'Submitting...';
          let idx = 0;
          ticker = setInterval(function() {{
            idx = (idx + 1) % frames.length;
            subtitle.textContent = frames[idx];
          }}, 1800);
        }});
        window.addEventListener('pageshow', function() {{
          if (ticker) clearInterval(ticker);
        }});
      }})();
    </script>
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
