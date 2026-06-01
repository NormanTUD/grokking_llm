# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "numpy",
#     "flask",
#     "plotly",
# ]
# ///
"""
Neural Network Binary Bisect Tool

Like git bisect but for neural network parameters. Enter a prompt,
then iteratively disable half the weights to find which parameter
groups (circuits) are responsible for the output.

Uses a web UI (Flask + Plotly) to visualize:
- Which weights are active vs disabled (zeroed)
- The model's output at each bisect step
- A tree of the bisection path showing which parameter groups matter

Supports: GPT-2 (small, medium, large), GPT-Neo, and other HuggingFace models.

Usage:
    uv run bisect_weights.py
"""

import sys
import os
import shutil
import subprocess

def _ensure_uv_run():
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return
    uv_path = shutil.which("uv")
    if uv_path is None:
        print("=" * 60)
        print("ERROR: This script must be run with `uv run` but `uv` was")
        print("not found on your system.")
        print("=" * 60)
        print()
        print("To install uv:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print()
        print("Then run:")
        print(f"  uv run {os.path.basename(__file__)}")
        print("=" * 60)
        sys.exit(1)
    script_path = os.path.abspath(__file__)
    extra_args = sys.argv[1:]
    cmd = [uv_path, "run", script_path] + extra_args
    print(f"[auto-restart] Re-launching with: {' '.join(cmd)}")
    env = os.environ.copy()
    env["_UV_RUN_ACTIVE"] = "1"
    if sys.platform == "win32":
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        os.execvpe(uv_path, cmd, env)

_ensure_uv_run()

import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template_string, request, jsonify
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Parameter Group Abstraction
# =============================================================================

@dataclass
class ParamGroup:
    group_id: str
    name: str
    param_name: str
    slice_start: int
    slice_end: int
    axis: int
    shape: tuple
    total_params: int
    enabled: bool = True
    importance_score: float = 0.0

    @property
    def num_params(self):
        size = self.slice_end - self.slice_start
        other_dims = 1
        for i, s in enumerate(self.shape):
            if i != self.axis:
                other_dims *= s
        return size * other_dims

@dataclass
class BisectSession:
    session_id: str
    prompt: str
    model_name: str
    baseline_output: str = ""
    baseline_logprob: float = 0.0
    current_output: str = ""
    current_logprob: float = 0.0
    param_groups: list = field(default_factory=list)
    history: list = field(default_factory=list)
    active_groups: list = field(default_factory=list)
    disabled_groups: list = field(default_factory=list)
    important_groups: list = field(default_factory=list)
    bisect_stack: list = field(default_factory=list)
    step: int = 0
    status: str = "idle"
    threshold: float = 0.1
    max_tokens: int = 20

# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.original_weights = {}
        self.param_groups = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    def load_model(self, model_name: str):
        print(f"[ModelManager] Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name

        self.original_weights = {}
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()

        self._build_param_groups()
        print(f"[ModelManager] Loaded {model_name} with {len(self.param_groups)} parameter groups")
        print(f"[ModelManager] Device: {self.device}")

    def _build_param_groups(self):
        self.param_groups = []
        group_idx = 0

        for name, param in self.model.named_parameters():
            if param.dim() < 2:
                pg = ParamGroup(
                    group_id=f"g_{group_idx}",
                    name=f"{name}",
                    param_name=name,
                    slice_start=0,
                    slice_end=param.shape[0],
                    axis=0,
                    shape=tuple(param.shape),
                    total_params=param.numel(),
                )
                self.param_groups.append(pg)
                group_idx += 1
            else:
                out_dim = param.shape[0]
                n_chunks = min(max(4, out_dim // 64), 16)
                chunk_size = math.ceil(out_dim / n_chunks)

                for i in range(n_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, out_dim)
                    if start >= out_dim:
                        break
                    pg = ParamGroup(
                        group_id=f"g_{group_idx}",
                        name=f"{name}[{start}:{end}]",
                        param_name=name,
                        slice_start=start,
                        slice_end=end,
                        axis=0,
                        shape=tuple(param.shape),
                        total_params=param.numel(),
                    )
                    self.param_groups.append(pg)
                    group_idx += 1

    def apply_mask(self, disabled_group_ids: set):
        with self._lock:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_weights[name])

            param_dict = dict(self.model.named_parameters())
            for pg in self.param_groups:
                if pg.group_id in disabled_group_ids:
                    param = param_dict[pg.param_name]
                    if pg.axis == 0:
                        param.data[pg.slice_start:pg.slice_end] = 0.0

    def restore_all(self):
        with self._lock:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_weights[name])

    def generate(self, prompt: str, max_new_tokens: int = 20) -> tuple:
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            if outputs.scores:
                total_logprob = 0.0
                for i, score in enumerate(outputs.scores):
                    if i < len(generated_ids):
                        log_probs = F.log_softmax(score[0], dim=-1)
                        token_logprob = log_probs[generated_ids[i]].item()
                        total_logprob += token_logprob
                avg_logprob = total_logprob / max(len(outputs.scores), 1)
            else:
                avg_logprob = 0.0

            return generated_text, avg_logprob

    def get_param_group_stats(self) -> dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        stats = {
            "total_params": total_params,
            "num_groups": len(self.param_groups),
            "layers": {},
        }
        for pg in self.param_groups:
            parts = pg.param_name.split(".")
            layer_name = ".".join(parts[:3]) if len(parts) > 3 else pg.param_name
            if layer_name not in stats["layers"]:
                stats["layers"][layer_name] = {"count": 0, "params": 0}
            stats["layers"][layer_name]["count"] += 1
            stats["layers"][layer_name]["params"] += pg.num_params
        return stats

# =============================================================================
# Bisect Engine
# =============================================================================

class BisectEngine:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.session: Optional[BisectSession] = None
        self._running = False
        self._step_results = []

    def start_session(self, prompt: str, threshold: float = 0.1, max_tokens: int = 20) -> BisectSession:
        session = BisectSession(
            session_id=str(uuid.uuid4())[:8],
            prompt=prompt,
            model_name=self.mm.model_name,
            threshold=threshold,
            max_tokens=max_tokens,
        )

        self.mm.restore_all()
        baseline_text, baseline_logprob = self.mm.generate(prompt, max_tokens)
        session.baseline_output = baseline_text
        session.baseline_logprob = baseline_logprob
        session.current_output = baseline_text
        session.current_logprob = baseline_logprob

        all_group_ids = [pg.group_id for pg in self.mm.param_groups]
        session.active_groups = list(all_group_ids)
        session.param_groups = list(self.mm.param_groups)

        session.bisect_stack = [all_group_ids]
        session.status = "ready"

        self.session = session
        self._step_results = []
        return session

    def bisect_step(self) -> dict:
        if self.session is None:
            return {"error": "No active session"}

        session = self.session

        if not session.bisect_stack:
            session.status = "complete"
            return {"status": "complete", "message": "Bisection complete!"}

        current_groups = session.bisect_stack.pop()

        if len(current_groups) <= 1:
            if current_groups:
                session.important_groups.append(current_groups[0])
            return {
                "status": "leaf",
                "message": f"Found important group: {current_groups[0] if current_groups else 'none'}",
                "group_id": current_groups[0] if current_groups else None,
                "step": session.step,
                "disabled_total": len(session.disabled_groups),
                "important_total": len(session.important_groups),
                "remaining_to_test": len(session.bisect_stack),
            }

        mid = len(current_groups) // 2
        first_half = current_groups[:mid]
        second_half = current_groups[mid:]

        session.step += 1

        # Test with first half disabled
        disabled_set = set(session.disabled_groups) | set(first_half)
        self.mm.apply_mask(disabled_set)
        text_without_first, logprob_without_first = self.mm.generate(session.prompt, session.max_tokens)

        # Test with second half disabled
        disabled_set = set(session.disabled_groups) | set(second_half)
        self.mm.apply_mask(disabled_set)
        text_without_second, logprob_without_second = self.mm.generate(session.prompt, session.max_tokens)

        baseline_lp = session.baseline_logprob
        delta_first = abs(baseline_lp - logprob_without_first)
        delta_second = abs(baseline_lp - logprob_without_second)

        text_match_without_first = (text_without_first.strip()[:50] == session.baseline_output.strip()[:50])
        text_match_without_second = (text_without_second.strip()[:50] == session.baseline_output.strip()[:50])

        threshold = session.threshold
        first_important = delta_first > threshold or not text_match_without_first
        second_important = delta_second > threshold or not text_match_without_second

        if first_important and len(first_half) > 1:
            session.bisect_stack.append(first_half)
        elif first_important and len(first_half) == 1:
            session.important_groups.append(first_half[0])

        if second_important and len(second_half) > 1:
            session.bisect_stack.append(second_half)
        elif second_important and len(second_half) == 1:
            session.important_groups.append(second_half[0])

        if not first_important:
            session.disabled_groups.extend(first_half)
        if not second_important:
            session.disabled_groups.extend(second_half)

        # Update current output
        self.mm.apply_mask(set(session.disabled_groups))
        current_text, current_logprob = self.mm.generate(session.prompt, session.max_tokens)
        session.current_output = current_text
        session.current_logprob = current_logprob

        result = {
            "status": "bisecting",
            "step": session.step,
            "total_groups_tested": len(current_groups),
            "first_half_size": len(first_half),
            "second_half_size": len(second_half),
            "without_first_half": {
                "text": text_without_first[:100],
                "logprob": logprob_without_first,
                "delta": delta_first,
                "text_matches": text_match_without_first,
            },
            "without_second_half": {
                "text": text_without_second[:100],
                "logprob": logprob_without_second,
                "delta": delta_second,
                "text_matches": text_match_without_second,
            },
            "first_important": first_important,
            "second_important": second_important,
            "disabled_total": len(session.disabled_groups),
            "important_total": len(session.important_groups),
            "remaining_to_test": len(session.bisect_stack),
            "current_output": current_text[:100],
            "current_logprob": current_logprob,
        }

        session.history.append(result)
        self._step_results.append(result)

        # Restore for next step
        self.mm.restore_all()

        return result

    def get_visualization_data(self) -> dict:
        if self.session is None:
            return {}

        session = self.session
        disabled_set = set(session.disabled_groups)
        important_set = set(session.important_groups)

        weight_map = []
        for pg in self.mm.param_groups:
            status = "active"
            if pg.group_id in disabled_set:
                status = "disabled"
            elif pg.group_id in important_set:
                status = "important"

            weight_map.append({
                "group_id": pg.group_id,
                "name": pg.name,
                "param_name": pg.param_name,
                "slice": f"[{pg.slice_start}:{pg.slice_end}]",
                "num_params": pg.num_params,
                "status": status,
            })

        return {
            "session_id": session.session_id,
            "prompt": session.prompt,
            "baseline_output": session.baseline_output,
            "current_output": session.current_output,
            "baseline_logprob": session.baseline_logprob,
            "current_logprob": session.current_logprob,
            "step": session.step,
            "status": session.status,
            "total_groups": len(self.mm.param_groups),
            "disabled_count": len(session.disabled_groups),
            "important_count": len(session.important_groups),
            "remaining_stack": len(session.bisect_stack),
            "weight_map": weight_map,
            "history": session.history,
            "threshold": session.threshold,
        }

# =============================================================================
# Flask Web Application
# =============================================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Binary Bisect Tool</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #161b22, #1f2937);
            border-bottom: 1px solid #30363d;
            padding: 16px 30px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 { font-size: 1.4em; color: #58a6ff; }
        .header .subtitle { color: #8b949e; font-size: 0.85em; }
        .container { max-width: 1800px; margin: 0 auto; padding: 15px; }
        .grid {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 15px;
            margin-top: 15px;
        }
        .panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .panel h2 {
            color: #58a6ff;
            font-size: 1em;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid #30363d;
        }
        label { display: block; color: #8b949e; font-size: 0.8em; margin-bottom: 3px; margin-top: 8px; }
        input[type="text"], input[type="number"], select, textarea {
            width: 100%;
            padding: 7px 10px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 5px;
            color: #c9d1d9;
            font-size: 0.85em;
        }
        textarea { resize: vertical; min-height: 50px; font-family: inherit; }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 5px;
            font-size: 0.85em;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.15s;
            margin-top: 8px;
            width: 100%;
        }
        .btn-primary { background: #238636; color: white; }
        .btn-primary:hover { background: #2ea043; }
        .btn-primary:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        .btn-bisect { background: #1f6feb; color: white; }
        .btn-bisect:hover { background: #388bfd; }
        .btn-bisect:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        .btn-auto { background: #8957e5; color: white; }
        .btn-auto:hover { background: #a371f7; }
        .btn-auto:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        .btn-danger { background: #da3633; color: white; }
        .btn-danger:hover { background: #f85149; }
        .output-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 5px;
            padding: 10px;
            margin: 6px 0;
            font-family: 'Fira Code', monospace;
            font-size: 0.8em;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 120px;
            overflow-y: auto;
            line-height: 1.4;
        }
        .output-box.baseline { border-left: 3px solid #238636; }
        .output-box.current { border-left: 3px solid #1f6feb; }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin: 8px 0;
        }
        .stat-item {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 8px;
            text-align: center;
        }
        .stat-item .number { font-size: 1.3em; font-weight: 700; color: #58a6ff; }
        .stat-item .label { font-size: 0.7em; color: #8b949e; margin-top: 2px; }
        .stat-item.red .number { color: #f85149; }
        .stat-item.green .number { color: #3fb950; }
        .stat-item.blue .number { color: #58a6ff; }
        .stat-item.purple .number { color: #a371f7; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #21262d;
            border-radius: 4px;
            overflow: hidden;
            margin: 8px 0;
        }
        .progress-bar .fill {
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 4px;
        }
        .progress-bar .fill.green { background: linear-gradient(90deg, #238636, #3fb950); }
        .progress-bar .fill.red { background: linear-gradient(90deg, #da3633, #f85149); }
        .progress-bar .fill.blue { background: linear-gradient(90deg, #1f6feb, #58a6ff); }
        .right-panel {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .viz-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        .viz-panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px;
            min-height: 300px;
        }
        .viz-panel.full { grid-column: 1 / -1; }
        .log-area {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 5px;
            padding: 10px;
            font-family: 'Fira Code', monospace;
            font-size: 0.75em;
            max-height: 200px;
            overflow-y: auto;
            white-space: pre-wrap;
            line-height: 1.5;
        }
        .log-entry { margin-bottom: 2px; }
        .log-entry.info { color: #58a6ff; }
        .log-entry.success { color: #3fb950; }
        .log-entry.warning { color: #d29922; }
        .log-entry.error { color: #f85149; }
        .log-entry.step { color: #a371f7; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 600;
        }
        .badge-green { background: #238636; color: white; }
        .badge-blue { background: #1f6feb; color: white; }
        .badge-yellow { background: #9e6a03; color: white; }
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin-right: 6px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .hidden { display: none !important; }
        .step-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 10px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 5px;
            margin: 6px 0;
            font-size: 0.8em;
        }
        .step-indicator .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #3fb950;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .step-indicator.idle .dot { background: #484f58; animation: none; }
        .step-indicator.running .dot { background: #3fb950; }
        .step-indicator.complete .dot { background: #58a6ff; animation: none; }
        @media (max-width: 1200px) {
            .grid { grid-template-columns: 1fr; }
            .viz-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Neural Network Binary Bisect</h1>
            <div class="subtitle">Find the circuits that matter by zeroing out weights</div>
        </div>
        <div>
            <span class="badge badge-yellow" id="model-badge">No Model</span>
            <span class="badge badge-blue" id="device-badge">-</span>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Left Panel -->
            <div>
                <div class="panel">
                    <h2>1. Load Model</h2>
                    <label>Model</label>
                    <select id="model-select">
                        <option value="distilgpt2">DistilGPT-2 (82M, fast)</option>
                        <option value="gpt2" selected>GPT-2 (124M)</option>
                        <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                        <option value="gpt2-large">GPT-2 Large (774M)</option>
                        <option value="EleutherAI/gpt-neo-125m">GPT-Neo 125M</option>
                    </select>
                    <button class="btn-primary" id="load-btn" onclick="loadModel()">Load Model</button>
                    <div id="load-status" class="step-indicator idle hidden">
                        <div class="dot"></div>
                        <span id="load-status-text">Loading...</span>
                    </div>
                </div>

                <div class="panel">
                    <h2>2. Configure</h2>
                    <label>Prompt</label>
                    <textarea id="prompt-input" rows="2">The capital of France is</textarea>
                    <label>Importance Threshold (lower = more sensitive)</label>
                    <input type="number" id="threshold-input" value="0.05" step="0.01" min="0.001" max="2.0">
                    <label>Max New Tokens</label>
                    <input type="number" id="max-tokens-input" value="15" step="1" min="1" max="50">
                    <button class="btn-primary" id="start-btn" onclick="startSession()" disabled>Start Bisection</button>
                </div>

                <div class="panel">
                    <h2>3. Bisect</h2>
                    <button class="btn-bisect" id="step-btn" onclick="bisectStep()" disabled>Single Step</button>
                    <button class="btn-auto" id="auto5-btn" onclick="autoBisect(5)" disabled>Auto 5 Steps</button>
                    <button class="btn-auto" id="auto20-btn" onclick="autoBisect(20)" disabled>Auto 20 Steps</button>
                    <button class="btn-auto" id="autoall-btn" onclick="autoBisect(200)" disabled>Run to Completion</button>
                    <button class="btn-danger" id="reset-btn" onclick="resetSession()" disabled>Reset</button>

                    <div id="run-indicator" class="step-indicator idle">
                        <div class="dot"></div>
                        <span id="run-status-text">Idle</span>
                    </div>

                    <div class="progress-bar">
                        <div class="fill green" id="progress-fill" style="width: 0%"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.7em; color:#8b949e; margin-top:2px;">
                        <span id="progress-label-left">0 classified</span>
                        <span id="progress-label-right">0%</span>
                    </div>

                    <div class="stats-grid">
                        <div class="stat-item blue">
                            <div class="number" id="stat-step">0</div>
                            <div class="label">Step</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="stat-total">0</div>
                            <div class="label">Total Groups</div>
                        </div>
                        <div class="stat-item red">
                            <div class="number" id="stat-disabled">0</div>
                            <div class="label">Disabled</div>
                        </div>
                        <div class="stat-item green">
                            <div class="number" id="stat-important">0</div>
                            <div class="label">Important</div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h2>Output Comparison</h2>
                    <label>Baseline (all weights active)</label>
                    <div class="output-box baseline" id="baseline-output">-</div>
                    <label>Current (with disabled weights)</label>
                    <div class="output-box current" id="current-output">-</div>
                    <label>Logprob Delta</label>
                    <div id="logprob-delta" style="font-size:0.85em; color:#d29922; margin-top:4px;">-</div>
                </div>

                <div class="panel">
                    <h2>Activity Log</h2>
                    <div class="log-area" id="log-area"></div>
                </div>
            </div>

            <!-- Right Panel: Visualizations -->
            <div class="right-panel">
                <div class="viz-row">
                    <div class="viz-panel full">
                        <div id="heatmap-plot" style="width:100%; height:450px;"></div>
                    </div>
                </div>
                <div class="viz-row">
                    <div class="viz-panel">
                        <div id="progress-plot" style="width:100%; height:350px;"></div>
                    </div>
                    <div class="viz-panel">
                        <div id="layers-plot" style="width:100%; height:350px;"></div>
                    </div>
                </div>
                <div class="viz-row">
                    <div class="viz-panel">
                        <div id="disabled-plot" style="width:100%; height:350px;"></div>
                    </div>
                    <div class="viz-panel">
                        <div id="important-plot" style="width:100%; height:350px;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let isRunning = false;
        let autoInterval = null;
        const plotlyConfig = {responsive: true, displayModeBar: false};
        const darkLayout = {
            paper_bgcolor: '#161b22',
            plot_bgcolor: '#0d1117',
            font: {color: '#c9d1d9', size: 11},
            margin: {l: 50, r: 20, t: 40, b: 40},
            xaxis: {gridcolor: '#21262d', zerolinecolor: '#30363d'},
            yaxis: {gridcolor: '#21262d', zerolinecolor: '#30363d'},
        };

        function addLog(msg, level) {
            level = level || 'info';
            const logArea = document.getElementById('log-area');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + level;
            const time = new Date().toLocaleTimeString();
            entry.textContent = '[' + time + '] ' + msg;
            logArea.appendChild(entry);
            logArea.scrollTop = logArea.scrollHeight;
        }

        function setRunStatus(status, text) {
            const indicator = document.getElementById('run-indicator');
            const statusText = document.getElementById('run-status-text');
            indicator.className = 'step-indicator ' + status;
            statusText.textContent = text;
        }

        function updateStats(data) {
            if (data.step !== undefined) document.getElementById('stat-step').textContent = data.step;
            if (data.total_groups !== undefined) document.getElementById('stat-total').textContent = data.total_groups;
            if (data.disabled_total !== undefined) document.getElementById('stat-disabled').textContent = data.disabled_total;
            else if (data.disabled_count !== undefined) document.getElementById('stat-disabled').textContent = data.disabled_count;
            if (data.important_total !== undefined) document.getElementById('stat-important').textContent = data.important_total;
            else if (data.important_count !== undefined) document.getElementById('stat-important').textContent = data.important_count;

            const total = parseInt(document.getElementById('stat-total').textContent) || 1;
            const disabled = parseInt(document.getElementById('stat-disabled').textContent) || 0;
            const important = parseInt(document.getElementById('stat-important').textContent) || 0;
            const classified = disabled + important;
            const pct = Math.round((classified / total) * 100);
            document.getElementById('progress-fill').style.width = pct + '%';
            document.getElementById('progress-label-left').textContent = classified + ' classified';
            document.getElementById('progress-label-right').textContent = pct + '%';
        }

        function updateLogprobDelta(baseline, current) {
            const delta = Math.abs(baseline - current);
            const el = document.getElementById('logprob-delta');
            el.textContent = 'Baseline: ' + baseline.toFixed(4) + ' | Current: ' + current.toFixed(4) + ' | Delta: ' + delta.toFixed(4);
            if (delta > 0.5) el.style.color = '#f85149';
            else if (delta > 0.1) el.style.color = '#d29922';
            else el.style.color = '#3fb950';
        }

        async function loadModel() {
            const btn = document.getElementById('load-btn');
            const modelName = document.getElementById('model-select').value;
            btn.disabled = true;
            btn.textContent = 'Loading...';

            const loadStatus = document.getElementById('load-status');
            loadStatus.classList.remove('hidden');
            loadStatus.className = 'step-indicator running';
            document.getElementById('load-status-text').textContent = 'Downloading & loading ' + modelName + '...';

            addLog('Loading model: ' + modelName + '...', 'info');

            try {
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_name: modelName})
                });
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                    loadStatus.className = 'step-indicator idle';
                    document.getElementById('load-status-text').textContent = 'Error: ' + data.error;
                } else {
                    addLog('Model loaded! ' + data.num_groups + ' parameter groups on ' + data.device, 'success');
                    loadStatus.className = 'step-indicator complete';
                    document.getElementById('load-status-text').textContent = modelName + ' | ' + data.num_groups + ' groups | ' + data.device;
                    document.getElementById('model-badge').textContent = modelName;
                    document.getElementById('model-badge').className = 'badge badge-green';
                    document.getElementById('device-badge').textContent = data.device.toUpperCase();
                    document.getElementById('start-btn').disabled = false;
                }
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
                loadStatus.className = 'step-indicator idle';
                document.getElementById('load-status-text').textContent = 'Network error';
            }

            btn.disabled = false;
            btn.textContent = 'Load Model';
        }

        async function startSession() {
            const btn = document.getElementById('start-btn');
            const prompt = document.getElementById('prompt-input').value;
            const threshold = parseFloat(document.getElementById('threshold-input').value);
            const maxTokens = parseInt(document.getElementById('max-tokens-input').value);
            btn.disabled = true;
            btn.textContent = 'Starting...';

            addLog('Starting bisection: "' + prompt + '" (threshold=' + threshold + ')', 'info');
            setRunStatus('running', 'Generating baseline...');

            try {
                const response = await fetch('/api/start_session', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt, threshold: threshold, max_tokens: maxTokens})
                });
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                    setRunStatus('idle', 'Error');
                } else {
                    addLog('Baseline: "' + data.baseline_output + '"', 'success');
                    document.getElementById('baseline-output').textContent = data.baseline_output;
                    document.getElementById('current-output').textContent = data.baseline_output;
                    document.getElementById('step-btn').disabled = false;
                    document.getElementById('auto5-btn').disabled = false;
                    document.getElementById('auto20-btn').disabled = false;
                    document.getElementById('autoall-btn').disabled = false;
                    document.getElementById('reset-btn').disabled = false;
                    updateStats(data);
                    setRunStatus('complete', 'Ready to bisect');
                    await refreshPlots();
                }
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
                setRunStatus('idle', 'Error');
            }

            btn.disabled = false;
            btn.textContent = 'Start Bisection';
        }

        async function bisectStep() {
            const btn = document.getElementById('step-btn');
            btn.disabled = true;
            setRunStatus('running', 'Bisecting...');

            try {
                const response = await fetch('/api/bisect_step', {method: 'POST'});
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                    setRunStatus('idle', 'Error');
                    btn.disabled = false;
                    return data;
                }

                if (data.status === 'complete') {
                    addLog('BISECTION COMPLETE! Found all important circuits.', 'success');
                    setRunStatus('complete', 'Complete!');
                    document.getElementById('step-btn').disabled = true;
                    document.getElementById('auto5-btn').disabled = true;
                    document.getElementById('auto20-btn').disabled = true;
                    document.getElementById('autoall-btn').disabled = true;
                    await refreshPlots();
                    return data;
                }

                if (data.status === 'leaf') {
                    addLog('Found important group: ' + (data.group_id || '?'), 'success');
                    updateStats(data);
                } else {
                    const msg = 'Step ' + data.step + ': tested ' + data.total_groups_tested + ' groups | ' +
                        'First half ' + (data.first_important ? 'IMPORTANT' : 'pruned') +
                        ', Second half ' + (data.second_important ? 'IMPORTANT' : 'pruned') +
                        ' | Disabled: ' + data.disabled_total + ' | Important: ' + data.important_total +
                        ' | Remaining: ' + data.remaining_to_test;
                    addLog(msg, data.first_important || data.second_important ? 'step' : 'warning');

                    if (data.current_output !== undefined) {
                        document.getElementById('current-output').textContent = data.current_output;
                    }
                    updateStats(data);
                }

                await refreshPlots();
                setRunStatus('complete', 'Step ' + (data.step || '?') + ' done');
                btn.disabled = false;
                return data;
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
                setRunStatus('idle', 'Error');
                btn.disabled = false;
                return {status: 'error'};
            }
        }

        async function autoBisect(steps) {
            if (isRunning) {
                isRunning = false;
                return;
            }
            isRunning = true;

            const btns = ['step-btn', 'auto5-btn', 'auto20-btn', 'autoall-btn'];
            btns.forEach(id => { document.getElementById(id).disabled = true; });

            addLog('Auto-bisecting up to ' + steps + ' steps...', 'info');
            setRunStatus('running', 'Auto-bisecting...');

            let completed = 0;
            for (let i = 0; i < steps && isRunning; i++) {
                const result = await bisectStep();
                completed++;

                if (!result || result.status === 'complete' || result.status === 'error') {
                    break;
                }

                // Small delay to let UI breathe
                await new Promise(r => setTimeout(r, 50));
            }

            isRunning = false;
            addLog('Auto-bisect finished after ' + completed + ' steps.', 'success');
            setRunStatus('complete', 'Auto-bisect done (' + completed + ' steps)');

            btns.forEach(id => { document.getElementById(id).disabled = false; });

            // Check if truly complete
            const vizResp = await fetch('/api/visualization_data');
            const vizData = await vizResp.json();
            if (vizData.status === 'complete' || vizData.remaining_stack === 0) {
                document.getElementById('step-btn').disabled = true;
                document.getElementById('auto5-btn').disabled = true;
                document.getElementById('auto20-btn').disabled = true;
                document.getElementById('autoall-btn').disabled = true;
                setRunStatus('complete', 'Bisection complete!');
            }
        }

        async function resetSession() {
            try {
                await fetch('/api/reset', {method: 'POST'});
                addLog('Session reset.', 'info');
                document.getElementById('baseline-output').textContent = '-';
                document.getElementById('current-output').textContent = '-';
                document.getElementById('logprob-delta').textContent = '-';
                document.getElementById('step-btn').disabled = true;
                document.getElementById('auto5-btn').disabled = true;
                document.getElementById('auto20-btn').disabled = true;
                document.getElementById('autoall-btn').disabled = true;
                document.getElementById('stat-step').textContent = '0';
                document.getElementById('stat-disabled').textContent = '0';
                document.getElementById('stat-important').textContent = '0';
                document.getElementById('progress-fill').style.width = '0%';
                document.getElementById('progress-label-left').textContent = '0 classified';
                document.getElementById('progress-label-right').textContent = '0%';
                setRunStatus('idle', 'Idle');
                clearAllPlots();
            } catch (e) {
                addLog('Error resetting: ' + e.message, 'error');
            }
        }

        function clearAllPlots() {
            const emptyData = [{x: [], y: [], type: 'scatter'}];
            const emptyLayout = Object.assign({}, darkLayout, {title: 'No data yet'});
            Plotly.react('heatmap-plot', emptyData, emptyLayout, plotlyConfig);
            Plotly.react('progress-plot', emptyData, emptyLayout, plotlyConfig);
            Plotly.react('layers-plot', emptyData, emptyLayout, plotlyConfig);
            Plotly.react('disabled-plot', emptyData, emptyLayout, plotlyConfig);
            Plotly.react('important-plot', emptyData, emptyLayout, plotlyConfig);
        }

        async function refreshPlots() {
            try {
                const response = await fetch('/api/plots');
                const plots = await response.json();

                if (plots.error) return;

                if (plots.heatmap) {
                    const hm = JSON.parse(plots.heatmap);
                    hm.layout = Object.assign({}, darkLayout, hm.layout || {});
                    Plotly.react('heatmap-plot', hm.data, hm.layout, plotlyConfig);
                }
                if (plots.progress) {
                    const pp = JSON.parse(plots.progress);
                    pp.layout = Object.assign({}, darkLayout, pp.layout || {});
                    Plotly.react('progress-plot', pp.data, pp.layout, plotlyConfig);
                }
                if (plots.layers) {
                    const lp = JSON.parse(plots.layers);
                    lp.layout = Object.assign({}, darkLayout, lp.layout || {});
                    Plotly.react('layers-plot', lp.data, lp.layout, plotlyConfig);
                }
                if (plots.disabled) {
                    const dp = JSON.parse(plots.disabled);
                    dp.layout = Object.assign({}, darkLayout, dp.layout || {});
                    Plotly.react('disabled-plot', dp.data, dp.layout, plotlyConfig);
                }
                if (plots.important) {
                    const ip = JSON.parse(plots.important);
                    ip.layout = Object.assign({}, darkLayout, ip.layout || {});
                    Plotly.react('important-plot', ip.data, ip.layout, plotlyConfig);
                }

                // Also update logprob delta
                const vizResp = await fetch('/api/visualization_data');
                const vizData = await vizResp.json();
                if (vizData && vizData.baseline_logprob !== undefined) {
                    updateLogprobDelta(vizData.baseline_logprob, vizData.current_logprob);
                    if (vizData.current_output) {
                        document.getElementById('current-output').textContent = vizData.current_output;
                    }
                }
            } catch (e) {
                // silently fail
            }
        }

        // Initialize empty plots on load
        window.addEventListener('load', function() {
            clearAllPlots();
        });
    </script>
</body>
</html>
"""

# =============================================================================
# Plot Generation Functions
# =============================================================================

def make_weight_heatmap(viz_data: dict) -> str:
    weight_map = viz_data.get("weight_map", [])
    if not weight_map:
        fig = go.Figure()
        fig.update_layout(title="No data yet - start a bisection session")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Group by layer
    layers = {}
    for wm in weight_map:
        parts = wm["param_name"].split(".")
        if len(parts) >= 3:
            layer = ".".join(parts[:3])
        else:
            layer = wm["param_name"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(wm)

    layer_names = list(layers.keys())
    max_groups = max(len(v) for v in layers.values()) if layers else 1

    z_values = []
    hover_texts = []
    for layer_name in layer_names:
        row = []
        hover_row = []
        for i in range(max_groups):
            if i < len(layers[layer_name]):
                wm = layers[layer_name][i]
                if wm["status"] == "disabled":
                    row.append(0)
                elif wm["status"] == "important":
                    row.append(2)
                else:
                    row.append(1)
                hover_row.append(
                    f"<b>{wm['name']}</b><br>"
                    f"Status: {wm['status']}<br>"
                    f"Params: {wm['num_params']:,}"
                )
            else:
                row.append(-0.5)
                hover_row.append("")
        z_values.append(row)
        hover_texts.append(hover_row)

    # Shorten layer names
    short_names = []
    for ln in layer_names:
        parts = ln.split(".")
        if len(parts) > 3:
            short_names.append(".".join(parts[-3:]))
        elif len(parts) > 2:
            short_names.append(".".join(parts[-2:]))
        else:
            short_names.append(ln)

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        y=short_names,
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.3, "#e74c3c"],
            [0.5, "#2ecc71"],
            [0.8, "#3498db"],
            [1.0, "#f39c12"],
        ],
        zmin=-0.5,
        zmax=2.5,
        showscale=False,
    ))

    total = viz_data.get('total_groups', 0)
    disabled = viz_data.get('disabled_count', 0)
    important = viz_data.get('important_count', 0)

    fig.update_layout(
        title=dict(
            text=f"Weight Map | Step {viz_data.get('step', 0)} | "
                 f"<span style='color:#e74c3c'>Disabled: {disabled}</span> | "
                 f"<span style='color:#2ecc71'>Active: {total - disabled - important}</span> | "
                 f"<span style='color:#3498db'>Important: {important}</span>",
            font=dict(size=12),
        ),
        height=max(350, min(len(layer_names) * 18, 600)),
        xaxis_title="Group Index",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=10, t=50, b=30),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_progress_plot(viz_data: dict) -> str:
    history = viz_data.get("history", [])
    if not history:
        fig = go.Figure()
        fig.update_layout(title="Bisection Progress (no steps yet)")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    steps = [h["step"] for h in history]
    disabled = [h["disabled_total"] for h in history]
    important = [h["important_total"] for h in history]
    remaining = [h["remaining_to_test"] for h in history]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Groups Classified", "Remaining to Test"),
                        vertical_spacing=0.2)

    fig.add_trace(go.Scatter(x=steps, y=disabled, mode="lines+markers",
                             name="Disabled", line=dict(color="#e74c3c", width=2),
                             marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=important, mode="lines+markers",
                             name="Important", line=dict(color="#3498db", width=2),
                             marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=remaining, mode="lines+markers",
                             name="Remaining", line=dict(color="#f39c12", width=2),
                             marker=dict(size=4)), row=2, col=1)

    fig.update_layout(
        height=350,
        title=dict(text="Bisection Progress", font=dict(size=12)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_layer_importance(viz_data: dict) -> str:
    weight_map = viz_data.get("weight_map", [])
    if not weight_map:
        fig = go.Figure()
        fig.update_layout(title="Layer Importance (no data)")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    layer_stats = {}
    for wm in weight_map:
        parts = wm["param_name"].split(".")
        if len(parts) >= 3:
            layer = ".".join(parts[:3])
        else:
            layer = wm["param_name"]
        if layer not in layer_stats:
            layer_stats[layer] = {"total": 0, "disabled": 0, "important": 0, "active": 0}
        layer_stats[layer]["total"] += 1
        if wm["status"] == "disabled":
            layer_stats[layer]["disabled"] += 1
        elif wm["status"] == "important":
            layer_stats[layer]["important"] += 1
        else:
            layer_stats[layer]["active"] += 1

    layers = list(layer_stats.keys())
    short_layers = []
    for l in layers:
        parts = l.split(".")
        if len(parts) > 2:
            short_layers.append(".".join(parts[-2:]))
        else:
            short_layers.append(l)

    disabled_pcts = []
    important_pcts = []
    active_pcts = []

    for l in layers:
        total = max(layer_stats[l]["total"], 1)
        disabled_pcts.append(layer_stats[l]["disabled"] / total * 100)
        important_pcts.append(layer_stats[l]["important"] / total * 100)
        active_pcts.append(layer_stats[l]["active"] / total * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Disabled", x=short_layers, y=disabled_pcts, marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name="Important", x=short_layers, y=important_pcts, marker_color="#3498db"))
    fig.add_trace(go.Bar(name="Untested", x=short_layers, y=active_pcts, marker_color="#2ecc71"))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Layer Status (%)", font=dict(size=12)),
        height=350,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=80),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_disabled_detail(viz_data: dict) -> str:
    weight_map = viz_data.get("weight_map", [])
    disabled = [wm for wm in weight_map if wm["status"] == "disabled"]

    if not disabled:
        fig = go.Figure()
        fig.update_layout(title="Disabled Groups (none yet)")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Show top 30 by param count
    disabled_sorted = sorted(disabled, key=lambda x: -x["num_params"])[:30]
    names = [d["name"][:40] for d in disabled_sorted]
    params = [d["num_params"] for d in disabled_sorted]

    fig = go.Figure(go.Bar(
        x=params,
        y=names,
        orientation="h",
        marker_color="#e74c3c",
        hovertemplate="<b>%{y}</b><br>Params: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Disabled Groups ({len(disabled)} total, top 30 by size)", font=dict(size=12)),
        height=350,
        margin=dict(l=250, r=10, t=40, b=30),
        xaxis_title="Parameters",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_important_detail(viz_data: dict) -> str:
    weight_map = viz_data.get("weight_map", [])
    important = [wm for wm in weight_map if wm["status"] == "important"]

    if not important:
        fig = go.Figure()
        fig.update_layout(title="Important Groups (none found yet)")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    important_sorted = sorted(important, key=lambda x: -x["num_params"])[:30]
    names = [d["name"][:40] for d in important_sorted]
    params = [d["num_params"] for d in important_sorted]

    fig = go.Figure(go.Bar(
        x=params,
        y=names,
        orientation="h",
        marker_color="#3498db",
        hovertemplate="<b>%{y}</b><br>Params: %{x:,}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"Important (Circuit) Groups ({len(important)} total, top 30)", font=dict(size=12)),
        height=350,
        margin=dict(l=250, r=10, t=40, b=30),
        xaxis_title="Parameters",
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)
model_manager = ModelManager()
bisect_engine = BisectEngine(model_manager)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/load_model", methods=["POST"])
def api_load_model():
    data = request.json
    model_name = data.get("model_name", "gpt2")
    try:
        model_manager.load_model(model_name)
        stats = model_manager.get_param_group_stats()
        return jsonify({
            "status": "ok",
            "model_name": model_name,
            "num_groups": stats["num_groups"],
            "total_params": stats["total_params"],
            "device": model_manager.device,
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/start_session", methods=["POST"])
def api_start_session():
    data = request.json
    prompt = data.get("prompt", "Hello")
    threshold = data.get("threshold", 0.05)
    max_tokens = data.get("max_tokens", 15)

    if model_manager.model is None:
        return jsonify({"error": "No model loaded. Load a model first."})

    try:
        session = bisect_engine.start_session(prompt, threshold, max_tokens)
        return jsonify({
            "status": "ok",
            "session_id": session.session_id,
            "baseline_output": session.baseline_output,
            "baseline_logprob": session.baseline_logprob,
            "total_groups": len(model_manager.param_groups),
            "step": 0,
            "disabled_count": 0,
            "important_count": 0,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/bisect_step", methods=["POST"])
def api_bisect_step():
    if bisect_engine.session is None:
        return jsonify({"error": "No active session. Start a session first."})

    try:
        result = bisect_engine.bisect_step()
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    if model_manager.model is not None:
        model_manager.restore_all()
    bisect_engine.session = None
    return jsonify({"status": "ok"})


@app.route("/api/visualization_data")
def api_visualization_data():
    try:
        viz_data = bisect_engine.get_visualization_data()
        if not viz_data:
            return jsonify({"error": "No session data available"})
        return jsonify(viz_data)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/plots")
def api_plots():
    try:
        viz_data = bisect_engine.get_visualization_data()
        if not viz_data:
            return jsonify({"error": "No data"})

        plots = {
            "heatmap": make_weight_heatmap(viz_data),
            "progress": make_progress_plot(viz_data),
            "layers": make_layer_importance(viz_data),
            "disabled": make_disabled_detail(viz_data),
            "important": make_important_detail(viz_data),
        }
        return jsonify(plots)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/api/model_stats")
def api_model_stats():
    if model_manager.model is None:
        return jsonify({"error": "No model loaded"})
    stats = model_manager.get_param_group_stats()
    return jsonify(stats)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    print("=" * 60)
    print("  Neural Network Binary Bisect Tool")
    print("  Like git bisect, but for neural network parameters")
    print("=" * 60)
    print()
    print("Starting web server...")
    print("Open your browser to: http://localhost:5000")
    print()
    print("Instructions:")
    print("  1. Select and load a model (e.g., gpt2 or distilgpt2)")
    print("  2. Enter a prompt")
    print("  3. Click 'Start Bisection' to get baseline output")
    print("  4. Click 'Single Step' or 'Auto' to find important weights")
    print("  5. Watch the visualizations update live!")
    print()
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
