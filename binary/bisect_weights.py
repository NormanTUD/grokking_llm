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

# =============================================================================
# Auto-restart under uv run if invoked directly with python3
# =============================================================================

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
import copy
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
    """Represents a group of parameters that can be enabled/disabled together."""
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
class BisectNode:
    """A node in the bisection tree."""
    node_id: str
    depth: int
    groups: list  # list of group_ids in this node
    enabled: bool = True
    output_text: str = ""
    output_logprob: float = 0.0
    loss_delta: float = 0.0
    children: list = field(default_factory=list)
    parent_id: Optional[str] = None
    verdict: Optional[str] = None  # "important", "unimportant", "testing"

@dataclass
class BisectSession:
    """Tracks the state of a bisection session."""
    session_id: str
    prompt: str
    model_name: str
    baseline_output: str = ""
    baseline_logprob: float = 0.0
    current_output: str = ""
    current_logprob: float = 0.0
    param_groups: list = field(default_factory=list)
    tree_root: Optional[BisectNode] = None
    history: list = field(default_factory=list)
    active_groups: list = field(default_factory=list)
    disabled_groups: list = field(default_factory=list)
    important_groups: list = field(default_factory=list)
    bisect_stack: list = field(default_factory=list)
    step: int = 0
    status: str = "idle"
    threshold: float = 0.1

# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages model loading, weight masking, and inference."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.original_weights = {}
        self.param_groups = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    def load_model(self, model_name: str):
        """Load a model from HuggingFace."""
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

        # Store original weights
        self.original_weights = {}
        for name, param in self.model.named_parameters():
            self.original_weights[name] = param.data.clone()

        # Build parameter groups
        self._build_param_groups()
        print(f"[ModelManager] Loaded {model_name} with {len(self.param_groups)} parameter groups")
        print(f"[ModelManager] Device: {self.device}")

    def _build_param_groups(self):
        """Build parameter groups by splitting each weight matrix into chunks."""
        self.param_groups = []
        group_idx = 0

        for name, param in self.model.named_parameters():
            if param.dim() < 2:
                # For biases and 1D params, treat as single group
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
                # For 2D+ params, split along the output dimension (axis 0)
                # into chunks of reasonable size
                out_dim = param.shape[0]
                # Target ~32-128 groups per layer for tractability
                n_chunks = min(max(4, out_dim // 64), 32)
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
        """Zero out parameters for disabled groups."""
        with self._lock:
            # First restore all weights
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_weights[name])

            # Then zero out disabled groups
            for pg in self.param_groups:
                if pg.group_id in disabled_group_ids:
                    param = dict(self.model.named_parameters())[pg.param_name]
                    if pg.axis == 0:
                        param.data[pg.slice_start:pg.slice_end] = 0.0
                    elif pg.axis == 1:
                        param.data[:, pg.slice_start:pg.slice_end] = 0.0

    def restore_all(self):
        """Restore all original weights."""
        with self._lock:
            for name, param in self.model.named_parameters():
                param.data.copy_(self.original_weights[name])

    def generate(self, prompt: str, max_new_tokens: int = 50) -> tuple:
        """Generate text and return (text, avg_logprob)."""
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

            # Compute average log probability of generated tokens
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

    def compute_loss_on_prompt(self, prompt: str) -> float:
        """Compute the model's loss (negative log likelihood) on the prompt itself."""
        with self._lock:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            if input_ids.shape[1] < 2:
                return 0.0

            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                return outputs.loss.item()

    def get_param_group_stats(self) -> dict:
        """Get statistics about parameter groups."""
        total_params = sum(p.numel() for p in self.model.parameters())
        stats = {
            "total_params": total_params,
            "num_groups": len(self.param_groups),
            "layers": {},
        }

        for pg in self.param_groups:
            # Extract layer name
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
    """Implements the binary search over parameter groups."""

    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.session: Optional[BisectSession] = None

    def start_session(self, prompt: str, threshold: float = 0.1) -> BisectSession:
        """Start a new bisection session."""
        session = BisectSession(
            session_id=str(uuid.uuid4())[:8],
            prompt=prompt,
            model_name=self.mm.model_name,
            threshold=threshold,
        )

        # Get baseline output with all weights active
        self.mm.restore_all()
        baseline_text, baseline_logprob = self.mm.generate(prompt)
        session.baseline_output = baseline_text
        session.baseline_logprob = baseline_logprob
        session.current_output = baseline_text
        session.current_logprob = baseline_logprob

        # Initialize all groups as active
        all_group_ids = [pg.group_id for pg in self.mm.param_groups]
        session.active_groups = list(all_group_ids)
        session.param_groups = list(self.mm.param_groups)

        # Create root node
        root = BisectNode(
            node_id="root",
            depth=0,
            groups=all_group_ids,
            enabled=True,
        )
        root.output_text = baseline_text
        root.output_logprob = baseline_logprob
        session.tree_root = root

        # Initialize bisect stack with the full set
        session.bisect_stack = [all_group_ids]
        session.status = "ready"

        self.session = session
        return session

    def bisect_step(self) -> dict:
        """Perform one bisection step: split current group in half, test each half."""
        if self.session is None:
            return {"error": "No active session"}

        session = self.session

        if not session.bisect_stack:
            session.status = "complete"
            return {"status": "complete", "message": "Bisection complete!"}

        # Get current group to bisect
        current_groups = session.bisect_stack.pop()

        if len(current_groups) <= 1:
            # Single group - mark as important (it survived bisection)
            if current_groups:
                session.important_groups.append(current_groups[0])
            return {
                "status": "leaf",
                "message": f"Found important group: {current_groups[0] if current_groups else 'none'}",
                "group_id": current_groups[0] if current_groups else None,
            }

        # Split into two halves
        mid = len(current_groups) // 2
        first_half = current_groups[:mid]
        second_half = current_groups[mid:]

        session.step += 1

        # Test with first half disabled
        disabled_set = set(session.disabled_groups) | set(first_half)
        self.mm.apply_mask(disabled_set)
        text_without_first, logprob_without_first = self.mm.generate(session.prompt)

        # Test with second half disabled
        disabled_set = set(session.disabled_groups) | set(second_half)
        self.mm.apply_mask(disabled_set)
        text_without_second, logprob_without_second = self.mm.generate(session.prompt)

        # Compute importance based on how much output changes
        # Lower logprob = more degradation = more important
        baseline_lp = session.baseline_logprob

        delta_first = abs(baseline_lp - logprob_without_first)
        delta_second = abs(baseline_lp - logprob_without_second)

        # Also check text similarity
        text_match_without_first = (text_without_first.strip() == session.baseline_output.strip())
        text_match_without_second = (text_without_second.strip() == session.baseline_output.strip())

        result = {
            "status": "bisecting",
            "step": session.step,
            "total_groups": len(current_groups),
            "first_half_size": len(first_half),
            "second_half_size": len(second_half),
            "without_first_half": {
                "text": text_without_first,
                "logprob": logprob_without_first,
                "delta": delta_first,
                "text_matches": text_match_without_first,
            },
            "without_second_half": {
                "text": text_without_second,
                "logprob": logprob_without_second,
                "delta": delta_second,
                "text_matches": text_match_without_second,
            },
        }

        # Decide which halves to keep investigating
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

        # If a half is NOT important, we can safely disable it
        if not first_important:
            session.disabled_groups.extend(first_half)
        if not second_important:
            session.disabled_groups.extend(second_half)

        # Update current output
        self.mm.apply_mask(set(session.disabled_groups))
        current_text, current_logprob = self.mm.generate(session.prompt)
        session.current_output = current_text
        session.current_logprob = current_logprob

        # Record in history
        session.history.append({
            "step": session.step,
            "action": "bisect",
            "groups_tested": len(current_groups),
            "first_important": first_important,
            "second_important": second_important,
            "disabled_count": len(session.disabled_groups),
            "important_count": len(session.important_groups),
            "current_output": current_text,
            "remaining_stack": len(session.bisect_stack),
        })

        result["first_important"] = first_important
        result["second_important"] = second_important
        result["disabled_total"] = len(session.disabled_groups)
        result["important_total"] = len(session.important_groups)
        result["remaining_to_test"] = len(session.bisect_stack)
        result["current_output"] = current_text

        # Restore for next step
        self.mm.restore_all()

        return result

    def auto_bisect(self, max_steps: int = 100) -> list:
        """Run bisection automatically until complete or max steps reached."""
        results = []
        for _ in range(max_steps):
            result = self.bisect_step()
            results.append(result)
            if result.get("status") in ("complete", "error"):
                break
        return results

    def get_visualization_data(self) -> dict:
        """Get data needed for visualization."""
        if self.session is None:
            return {}

        session = self.session

        # Build weight map data
        weight_map = []
        disabled_set = set(session.disabled_groups)
        important_set = set(session.important_groups)

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
# Visualization Generators
# =============================================================================

def make_weight_heatmap(viz_data: dict) -> str:
    """Create a heatmap showing which weight groups are active/disabled/important."""
    weight_map = viz_data.get("weight_map", [])
    if not weight_map:
        fig = go.Figure()
        fig.update_layout(title="No data yet")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Group by layer
    layers = {}
    for wm in weight_map:
        layer = wm["param_name"].rsplit(".", 1)[0] if "." in wm["param_name"] else wm["param_name"]
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(wm)

    # Create a grid: rows = layers, columns = groups within layer
    layer_names = list(layers.keys())
    max_groups_per_layer = max(len(v) for v in layers.values())

    # Build color matrix
    z_values = []
    hover_texts = []
    for layer_name in layer_names:
        row = []
        hover_row = []
        for i in range(max_groups_per_layer):
            if i < len(layers[layer_name]):
                wm = layers[layer_name][i]
                if wm["status"] == "disabled":
                    row.append(0)
                elif wm["status"] == "important":
                    row.append(2)
                else:
                    row.append(1)
                hover_row.append(f"{wm['name']}<br>Status: {wm['status']}<br>Params: {wm['num_params']}")
            else:
                row.append(-1)
                hover_row.append("")
        z_values.append(row)
        hover_texts.append(hover_row)

    # Truncate layer names for display
    short_names = []
    for ln in layer_names:
        parts = ln.split(".")
        if len(parts) > 3:
            short_names.append(".".join(parts[-3:]))
        else:
            short_names.append(ln)

    colorscale = [
        [0.0, "rgba(50,50,50,0.3)"],    # -1: empty
        [0.25, "rgb(220,50,50)"],         # 0: disabled (red)
        [0.5, "rgb(50,150,50)"],          # 1: active (green)
        [0.75, "rgb(50,100,220)"],        # 2: important (blue)
        [1.0, "rgb(255,215,0)"],          # padding
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        y=short_names,
        hovertext=hover_texts,
        hoverinfo="text",
        colorscale=[
            [0.0, "rgb(220,50,50)"],
            [0.5, "rgb(100,200,100)"],
            [1.0, "rgb(50,100,220)"],
        ],
        zmin=0,
        zmax=2,
        showscale=False,
    ))

    fig.update_layout(
        title=f"Weight Groups: Red=Disabled, Green=Active, Blue=Important<br>"
              f"Step {viz_data['step']} | "
              f"Disabled: {viz_data['disabled_count']} | "
              f"Important: {viz_data['important_count']}",
        height=max(400, len(layer_names) * 20),
        xaxis_title="Group Index within Layer",
        yaxis_title="Layer",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=200),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_bisect_progress_plot(viz_data: dict) -> str:
    """Create a plot showing bisection progress over steps."""
    history = viz_data.get("history", [])
    if not history:
        fig = go.Figure()
        fig.update_layout(title="No bisection steps yet")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    steps = [h["step"] for h in history]
    disabled_counts = [h["disabled_count"] for h in history]
    important_counts = [h["important_count"] for h in history]
    remaining = [h["remaining_stack"] for h in history]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Groups Classified Over Time", "Remaining to Test"),
                        vertical_spacing=0.15)

    fig.add_trace(go.Scatter(x=steps, y=disabled_counts, mode="lines+markers",
                             name="Disabled (unimportant)", line=dict(color="red")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=important_counts, mode="lines+markers",
                             name="Important", line=dict(color="blue")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=steps, y=remaining, mode="lines+markers",
                             name="Remaining stack", line=dict(color="orange")),
                  row=2, col=1)

    fig.update_layout(height=500, title="Bisection Progress")
    fig.update_xaxes(title_text="Step", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_layer_importance_plot(viz_data: dict) -> str:
    """Create a bar chart showing importance by layer."""
    weight_map = viz_data.get("weight_map", [])
    if not weight_map:
        fig = go.Figure()
        fig.update_layout(title="No data yet")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Aggregate by layer
    layer_stats = {}
    for wm in weight_map:
        parts = wm["param_name"].split(".")
        # Get a reasonable layer grouping
        if len(parts) >= 3:
            layer = ".".join(parts[:3])
        else:
            layer = wm["param_name"]

        if layer not in layer_stats:
            layer_stats[layer] = {"total": 0, "disabled": 0, "important": 0, "active": 0}
        layer_stats[layer]["total"] += 1
        layer_stats[layer][wm["status"]] += 1

    layers = list(layer_stats.keys())
    # Shorten names
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
        total = layer_stats[l]["total"]
        if total == 0:
            disabled_pcts.append(0)
            important_pcts.append(0)
            active_pcts.append(0)
        else:
            disabled_pcts.append(layer_stats[l]["disabled"] / total * 100)
            important_pcts.append(layer_stats[l]["important"] / total * 100)
            active_pcts.append(layer_stats[l].get("active", 0) / total * 100)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Disabled", x=short_layers, y=disabled_pcts, marker_color="red"))
    fig.add_trace(go.Bar(name="Important", x=short_layers, y=important_pcts, marker_color="blue"))
    fig.add_trace(go.Bar(name="Active (untested)", x=short_layers, y=active_pcts, marker_color="lightgreen"))

    fig.update_layout(
        barmode="stack",
        title="Layer-wise Parameter Group Status (%)",
        xaxis_title="Layer",
        yaxis_title="Percentage",
        height=400,
        xaxis_tickangle=-45,
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_disabled_weights_detail(viz_data: dict) -> str:
    """Show details of disabled weight groups."""
    weight_map = viz_data.get("weight_map", [])
    disabled = [wm for wm in weight_map if wm["status"] == "disabled"]

    if not disabled:
        fig = go.Figure()
        fig.update_layout(title="No disabled groups yet")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Show as a table-like visualization
    names = [d["name"][:50] for d in disabled[:50]]  # limit for display
    params = [d["num_params"] for d in disabled[:50]]

    fig = go.Figure(go.Bar(
        x=params,
        y=names,
        orientation="h",
        marker_color="rgb(220,50,50)",
        hovertemplate="%{y}<br>Params: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Disabled (Unimportant) Groups ({len(disabled)} total, showing first 50)",
        xaxis_title="Number of Parameters",
        height=max(300, len(names) * 20),
        margin=dict(l=300),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def make_important_weights_detail(viz_data: dict) -> str:
    """Show details of important weight groups."""
    weight_map = viz_data.get("weight_map", [])
    important = [wm for wm in weight_map if wm["status"] == "important"]

    if not important:
        fig = go.Figure()
        fig.update_layout(title="No important groups identified yet")
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    names = [d["name"][:50] for d in important[:50]]
    params = [d["num_params"] for d in important[:50]]

    fig = go.Figure(go.Bar(
        x=params,
        y=names,
        orientation="h",
        marker_color="rgb(50,100,220)",
        hovertemplate="%{y}<br>Params: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Important (Circuit) Groups ({len(important)} total, showing first 50)",
        xaxis_title="Number of Parameters",
        height=max(300, len(names) * 20),
        margin=dict(l=300),
    )

    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# =============================================================================
# Flask Web Application
# =============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Binary Bisect Tool</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
            padding: 20px 40px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 1.5em;
            color: #58a6ff;
        }
        .header .subtitle {
            color: #8b949e;
            font-size: 0.9em;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        .grid {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        .panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
        }
        .panel h2 {
            color: #58a6ff;
            font-size: 1.1em;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #30363d;
        }
        .panel h3 {
            color: #79c0ff;
            font-size: 0.95em;
            margin: 15px 0 8px 0;
        }
        label {
            display: block;
            color: #8b949e;
            font-size: 0.85em;
            margin-bottom: 4px;
            margin-top: 10px;
        }
        input[type="text"], input[type="number"], select {
            width: 100%;
            padding: 8px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 0.9em;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #58a6ff;
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
        }
        textarea {
            width: 100%;
            padding: 10px 12px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 0.9em;
            resize: vertical;
            min-height: 60px;
            font-family: inherit;
        }
        textarea:focus {
            outline: none;
            border-color: #58a6ff;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 0.9em;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
            margin-top: 10px;
            width: 100%;
        }
        .btn-primary {
            background: #238636;
            color: white;
        }
        .btn-primary:hover { background: #2ea043; }
        .btn-primary:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }
        .btn-secondary {
            background: #21262d;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }
        .btn-secondary:hover { background: #30363d; }
        .btn-danger {
            background: #da3633;
            color: white;
        }
        .btn-danger:hover { background: #f85149; }
        .btn-bisect {
            background: #1f6feb;
            color: white;
        }
        .btn-bisect:hover { background: #388bfd; }
        .btn-auto {
            background: #8957e5;
            color: white;
        }
        .btn-auto:hover { background: #a371f7; }
        .status-bar {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 15px;
            margin: 10px 0;
            font-size: 0.85em;
        }
        .status-bar .label { color: #8b949e; }
        .status-bar .value { color: #58a6ff; font-weight: 600; }
        .output-box {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            margin: 10px 0;
            font-family: 'Fira Code', 'Cascadia Code', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        .output-box.baseline { border-left: 3px solid #238636; }
        .output-box.current { border-left: 3px solid #1f6feb; }
        .output-box.diff { border-left: 3px solid #da3633; }
        .viz-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .viz-panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            min-height: 400px;
        }
        .viz-panel.full-width {
            grid-column: 1 / -1;
        }
        .log-area {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 12px;
            font-family: 'Fira Code', monospace;
            font-size: 0.8em;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        .log-entry { margin-bottom: 4px; }
        .log-entry.info { color: #58a6ff; }
        .log-entry.success { color: #3fb950; }
        .log-entry.warning { color: #d29922; }
        .log-entry.error { color: #f85149; }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 10px 0;
        }
        .stat-item {
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 4px;
            padding: 8px;
            text-align: center;
        }
        .stat-item .number {
            font-size: 1.4em;
            font-weight: 700;
            color: #58a6ff;
        }
        .stat-item .label {
            font-size: 0.75em;
            color: #8b949e;
            margin-top: 2px;
        }
        .progress-bar {
            width: 100%;
            height: 6px;
            background: #21262d;
            border-radius: 3px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-bar .fill {
            height: 100%;
            background: linear-gradient(90deg, #238636, #3fb950);
            transition: width 0.3s ease;
        }
        .tab-bar {
            display: flex;
            gap: 2px;
            margin-bottom: 15px;
            background: #0d1117;
            border-radius: 8px;
            padding: 4px;
        }
        .tab-btn {
            padding: 8px 16px;
            border: none;
            background: transparent;
            color: #8b949e;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            font-weight: 500;
            width: auto;
            margin: 0;
        }
        .tab-btn.active {
            background: #21262d;
            color: #58a6ff;
        }
        .tab-btn:hover:not(.active) { color: #c9d1d9; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #30363d;
            border-top-color: #58a6ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .hidden { display: none !important; }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            font-weight: 600;
        }
        .badge-green { background: #238636; color: white; }
        .badge-red { background: #da3633; color: white; }
        .badge-blue { background: #1f6feb; color: white; }
        .badge-yellow { background: #9e6a03; color: white; }
        @media (max-width: 1200px) {
            .grid { grid-template-columns: 1fr; }
            .viz-container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Neural Network Binary Bisect Tool</h1>
            <div class="subtitle">Like git bisect, but for neural network parameters. Find the circuits that matter.</div>
        </div>
        <div>
            <span class="badge badge-blue" id="model-badge">No Model</span>
            <span class="badge badge-yellow" id="device-badge">CPU</span>
        </div>
    </div>

    <div class="container">
        <div class="grid">
            <!-- Left Panel: Controls -->
            <div>
                <div class="panel">
                    <h2>Model Setup</h2>
                    <label for="model-select">Model</label>
                    <select id="model-select">
                        <option value="gpt2">GPT-2 (124M)</option>
                        <option value="gpt2-medium">GPT-2 Medium (355M)</option>
                        <option value="gpt2-large">GPT-2 Large (774M)</option>
                        <option value="gpt2-xl">GPT-2 XL (1.5B)</option>
                        <option value="EleutherAI/gpt-neo-125m">GPT-Neo 125M</option>
                        <option value="EleutherAI/gpt-neo-1.3B">GPT-Neo 1.3B</option>
                        <option value="distilgpt2">DistilGPT-2 (82M)</option>
                    </select>
                    <button class="btn-primary" id="load-model-btn" onclick="loadModel()">Load Model</button>
                    <div id="model-status" class="status-bar hidden">
                        <span class="label">Status:</span> <span class="value" id="model-status-text"></span>
                    </div>
                </div>

                <div class="panel" style="margin-top: 15px;">
                    <h2>Prompt</h2>
                    <textarea id="prompt-input" placeholder="Enter your prompt here...">The capital of France is</textarea>
                    <label for="threshold-input">Importance Threshold</label>
                    <input type="number" id="threshold-input" value="0.1" step="0.01" min="0.01" max="1.0">
                    <label for="max-tokens-input">Max New Tokens</label>
                    <input type="number" id="max-tokens-input" value="20" step="1" min="1" max="100">
                    <button class="btn-primary" id="start-btn" onclick="startSession()" disabled>Start Bisection</button>
                </div>

                <div class="panel" style="margin-top: 15px;">
                    <h2>Bisection Controls</h2>
                    <button class="btn-bisect" id="step-btn" onclick="bisectStep()" disabled>Bisect Step</button>
                    <button class="btn-auto" id="auto-btn" onclick="autoBisect()" disabled>Auto Bisect (10 steps)</button>
                    <button class="btn-danger" id="reset-btn" onclick="resetSession()" disabled>Reset</button>

                    <div class="progress-bar" id="progress-bar">
                        <div class="fill" id="progress-fill" style="width: 0%"></div>
                    </div>

                    <div class="stats-grid" id="stats-grid">
                        <div class="stat-item">
                            <div class="number" id="stat-step">0</div>
                            <div class="label">Step</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="stat-total">0</div>
                            <div class="label">Total Groups</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="stat-disabled">0</div>
                            <div class="label">Disabled</div>
                        </div>
                        <div class="stat-item">
                            <div class="number" id="stat-important">0</div>
                            <div class="label">Important</div>
                        </div>
                    </div>
                </div>

                <div class="panel" style="margin-top: 15px;">
                    <h2>Output Comparison</h2>
                    <h3>Baseline (all weights active)</h3>
                    <div class="output-box baseline" id="baseline-output">-</div>
                    <h3>Current (with disabled weights)</h3>
                    <div class="output-box current" id="current-output">-</div>
                </div>
            </div>

            <!-- Right Panel: Visualizations -->
            <div>
                <div class="tab-bar">
                    <button class="tab-btn active" onclick="switchTab('heatmap')">Weight Map</button>
                    <button class="tab-btn" onclick="switchTab('progress')">Progress</button>
                    <button class="tab-btn" onclick="switchTab('layers')">Layer Importance</button>
                    <button class="tab-btn" onclick="switchTab('disabled')">Disabled Weights</button>
                    <button class="tab-btn" onclick="switchTab('important')">Important Weights</button>
                    <button class="tab-btn" onclick="switchTab('log')">Log</button>
                </div>

                <div class="tab-content active" id="tab-heatmap">
                    <div class="viz-panel full-width">
                        <div id="heatmap-plot" style="width:100%; min-height:500px;"></div>
                    </div>
                </div>

                <div class="tab-content" id="tab-progress">
                    <div class="viz-panel full-width">
                        <div id="progress-plot" style="width:100%; min-height:500px;"></div>
                    </div>
                </div>

                <div class="tab-content" id="tab-layers">
                    <div class="viz-panel full-width">
                        <div id="layers-plot" style="width:100%; min-height:500px;"></div>
                    </div>
                </div>

                <div class="tab-content" id="tab-disabled">
                    <div class="viz-panel full-width">
                        <div id="disabled-plot" style="width:100%; min-height:500px;"></div>
                    </div>
                </div>

                <div class="tab-content" id="tab-important">
                    <div class="viz-panel full-width">
                        <div id="important-plot" style="width:100%; min-height:500px;"></div>
                    </div>
                </div>

                <div class="tab-content" id="tab-log">
                    <div class="viz-panel full-width">
                        <div class="log-area" id="log-area"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentTab = 'heatmap';
        let isRunning = false;

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById('tab-' + tabName).classList.add('active');
            event.target.classList.add('active');
            currentTab = tabName;
        }

        function addLog(msg, level) {
            level = level || 'info';
            const logArea = document.getElementById('log-area');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + level;
            entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
            logArea.appendChild(entry);
            logArea.scrollTop = logArea.scrollHeight;
        }

        function setLoading(btn, loading) {
            if (loading) {
                btn.disabled = true;
                btn.dataset.originalText = btn.textContent;
                btn.innerHTML = '<span class="spinner"></span>Loading...';
            } else {
                btn.disabled = false;
                btn.textContent = btn.dataset.originalText || btn.textContent;
            }
        }

        async function loadModel() {
            const btn = document.getElementById('load-model-btn');
            const modelName = document.getElementById('model-select').value;
            setLoading(btn, true);
            addLog('Loading model: ' + modelName + '...', 'info');

            document.getElementById('model-status').classList.remove('hidden');
            document.getElementById('model-status-text').textContent = 'Loading...';

            try {
                const response = await fetch('/api/load_model', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_name: modelName})
                });
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                    document.getElementById('model-status-text').textContent = 'Error: ' + data.error;
                } else {
                    addLog('Model loaded: ' + data.model_name + ' (' + data.num_groups + ' parameter groups)', 'success');
                    document.getElementById('model-status-text').textContent =
                        data.model_name + ' | ' + data.num_groups + ' groups | ' + data.device;
                    document.getElementById('model-badge').textContent = modelName;
                    document.getElementById('model-badge').className = 'badge badge-green';
                    document.getElementById('device-badge').textContent = data.device.toUpperCase();
                    document.getElementById('start-btn').disabled = false;
                }
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
                document.getElementById('model-status-text').textContent = 'Network error';
            }

            setLoading(btn, false);
        }

        async function startSession() {
            const btn = document.getElementById('start-btn');
            const prompt = document.getElementById('prompt-input').value;
            const threshold = parseFloat(document.getElementById('threshold-input').value);
            setLoading(btn, true);
            addLog('Starting bisection session with prompt: "' + prompt + '"', 'info');

            try {
                const response = await fetch('/api/start_session', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt: prompt, threshold: threshold})
                });
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                } else {
                    addLog('Session started. Baseline output: "' + data.baseline_output + '"', 'success');
                    document.getElementById('baseline-output').textContent = data.baseline_output;
                    document.getElementById('current-output').textContent = data.baseline_output;
                    document.getElementById('step-btn').disabled = false;
                    document.getElementById('auto-btn').disabled = false;
                    document.getElementById('reset-btn').disabled = false;
                    updateStats(data);
                    updateVisualizations();
                }
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
            }

            setLoading(btn, false);
        }

        async function bisectStep() {
            const btn = document.getElementById('step-btn');
            btn.disabled = true;
            addLog('Running bisect step...', 'info');

            try {
                const response = await fetch('/api/bisect_step', {method: 'POST'});
                const data = await response.json();

                if (data.error) {
                    addLog('Error: ' + data.error, 'error');
                } else if (data.status === 'complete') {
                    addLog('Bisection COMPLETE!', 'success');
                    document.getElementById('step-btn').disabled = true;
                    document.getElementById('auto-btn').disabled = true;
                } else if (data.status === 'leaf') {
                    addLog('Found leaf: ' + data.message, 'success');
                } else {
                    const msg = 'Step ' + data.step + ': ' +
                        'First half ' + (data.first_important ? 'IMPORTANT' : 'unimportant') +
                        ', Second half ' + (data.second_important ? 'IMPORTANT' : 'unimportant') +
                        ' | Disabled: ' + data.disabled_total + ' | Important: ' + data.important_total;
                    addLog(msg, data.first_important || data.second_important ? 'warning' : 'info');

                    if (data.current_output) {
                        document.getElementById('current-output').textContent = data.current_output;
                    }
                }

                updateVisualizations();
            } catch (e) {
                addLog('Network error: ' + e.message, 'error');
            }

            btn.disabled = false;
        }

        async function autoBisect() {
            const btn = document.getElementById('auto-btn');
            btn.disabled = true;
            isRunning = true;
            addLog('Starting auto-bisect (10 steps)...', 'info');

            for (let i = 0; i < 10 && isRunning; i++) {
                try {
                    const response = await fetch('/api/bisect_step', {method: 'POST'});
                    const data = await response.json();

                    if (data.error) {
                        addLog('Error: ' + data.error, 'error');
                        break;
                    } else if (data.status === 'complete') {
                        addLog('Bisection COMPLETE!', 'success');
                        document.getElementById('step-btn').disabled = true;
                        document.getElementById('auto-btn').disabled = true;
                        break;
                    } else if (data.status === 'leaf') {
                        addLog('Found leaf: ' + data.message, 'success');
                    } else {
                        const msg = 'Step ' + data.step + ': Disabled=' + data.disabled_total +
                            ' Important=' + data.important_total + ' Remaining=' + data.remaining_to_test;
                        addLog(msg, 'info');
                        if (data.current_output) {
                            document.getElementById('current-output').textContent = data.current_output;
                        }
                    }

                    updateVisualizations();
                    await new Promise(r => setTimeout(r, 100));
                } catch (e) {
                    addLog('Network error: ' + e.message, 'error');
                    break;
                }
            }

            isRunning = false;
            btn.disabled = false;
        }

        async function resetSession() {
            try {
                await fetch('/api/reset', {method: 'POST'});
                addLog('Session reset.', 'info');
                document.getElementById('baseline-output').textContent = '-';
                document.getElementById('current-output').textContent = '-';
                document.getElementById('step-btn').disabled = true;
                document.getElementById('auto-btn').disabled = true;
                document.getElementById('stat-step').textContent = '0';
                document.getElementById('stat-disabled').textContent = '0';
                document.getElementById('stat-important').textContent = '0';
                document.getElementById('progress-fill').style.width = '0%';
                clearPlots();
            } catch (e) {
                addLog('Error resetting: ' + e.message, 'error');
            }
        }

        function updateStats(data) {
            if (data.step !== undefined) document.getElementById('stat-step').textContent = data.step;
            if (data.total_groups !== undefined) document.getElementById('stat-total').textContent = data.total_groups;
            if (data.disabled_count !== undefined) document.getElementById('stat-disabled').textContent = data.disabled_count;
            if (data.important_count !== undefined) document.getElementById('stat-important').textContent = data.important_count;

            if (data.total_groups && data.disabled_count !== undefined && data.important_count !== undefined) {
                const classified = data.disabled_count + data.important_count;
                const pct = (classified / data.total_groups) * 100;
                document.getElementById('progress-fill').style.width = pct + '%';
            }
        }

        async function updateVisualizations() {
            try {
                const response = await fetch('/api/visualization_data');
                const data = await response.json();

                if (data.error) return;

                updateStats(data);

                if (data.baseline_output) {
                    document.getElementById('baseline-output').textContent = data.baseline_output;
                }
                if (data.current_output) {
                    document.getElementById('current-output').textContent = data.current_output;
                }

                // Update plots
                const plotResponse = await fetch('/api/plots');
                const plots = await plotResponse.json();

                if (plots.heatmap) {
                    Plotly.react('heatmap-plot', JSON.parse(plots.heatmap).data, JSON.parse(plots.heatmap).layout);
                }
                if (plots.progress) {
                    Plotly.react('progress-plot', JSON.parse(plots.progress).data, JSON.parse(plots.progress).layout);
                }
                if (plots.layers) {
                    Plotly.react('layers-plot', JSON.parse(plots.layers).data, JSON.parse(plots.layers).layout);
                }
                if (plots.disabled) {
                    Plotly.react('disabled-plot', JSON.parse(plots.disabled).data, JSON.parse(plots.disabled).layout);
                }
                if (plots.important) {
                    Plotly.react('important-plot', JSON.parse(plots.important).data, JSON.parse(plots.important).layout);
                }
            } catch (e) {
                // Silently fail on visualization updates
            }
        }

        function clearPlots() {
            ['heatmap-plot', 'progress-plot', 'layers-plot', 'disabled-plot', 'important-plot'].forEach(id => {
                Plotly.purge(id);
            });
        }

        // Initialize empty plots
        window.addEventListener('load', function() {
            const emptyLayout = {
                paper_bgcolor: '#161b22',
                plot_bgcolor: '#0d1117',
                font: {color: '#c9d1d9'},
                title: 'Load a model and start a session to see visualizations',
            };
            ['heatmap-plot', 'progress-plot', 'layers-plot', 'disabled-plot', 'important-plot'].forEach(id => {
                Plotly.newPlot(id, [], emptyLayout);
            });
        });
    </script>
</body>
</html>
"""

# =============================================================================
# Flask App Setup
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
            "layers": len(stats["layers"]),
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/start_session", methods=["POST"])
def api_start_session():
    data = request.json
    prompt = data.get("prompt", "Hello")
    threshold = data.get("threshold", 0.1)

    if model_manager.model is None:
        return jsonify({"error": "No model loaded. Load a model first."})

    try:
        session = bisect_engine.start_session(prompt, threshold)
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
        return jsonify({"error": str(e)})

@app.route("/api/bisect_step", methods=["POST"])
def api_bisect_step():
    if bisect_engine.session is None:
        return jsonify({"error": "No active session. Start a session first."})

    try:
        result = bisect_engine.bisect_step()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/auto_bisect", methods=["POST"])
def api_auto_bisect():
    data = request.json or {}
    max_steps = data.get("max_steps", 10)

    if bisect_engine.session is None:
        return jsonify({"error": "No active session. Start a session first."})

    try:
        results = bisect_engine.auto_bisect(max_steps=max_steps)
        return jsonify({"results": results})
    except Exception as e:
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
            "progress": make_bisect_progress_plot(viz_data),
            "layers": make_layer_importance_plot(viz_data),
            "disabled": make_disabled_weights_detail(viz_data),
            "important": make_important_weights_detail(viz_data),
        }
        return jsonify(plots)
    except Exception as e:
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
    print("  1. Select and load a model (e.g., gpt2)")
    print("  2. Enter a prompt")
    print("  3. Click 'Start Bisection' to get baseline output")
    print("  4. Click 'Bisect Step' or 'Auto Bisect' to find important weights")
    print("  5. Watch the visualizations update in real-time!")
    print()
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
