#!/usr/bin/env python3
"""
LLM Layer Sonification - Hear the neural network think!
Loads a transformer model, extracts layer activations during text generation,
converts them to audio, and displays everything in a browser interface.
"""

import argparse
import json
import os
import threading
import wave
import struct
import math
import tempfile
import base64
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from flask import Flask, render_template_string, jsonify, request, send_file

# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Layer Sonification - Hear the model think!"
    )
    parser.add_argument(
        "--model", type=str, default="gpt2",
        choices=["gpt2", "gpt2-medium", "distilgpt2"],
        help="Model to use (default: gpt2)"
    )
    parser.add_argument(
        "--prompt", type=str, default="The meaning of life is",
        help="Initial prompt for text generation"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=22050,
        help="Audio sample rate in Hz (default: 22050)"
    )
    parser.add_argument(
        "--duration-per-layer", type=float, default=0.15,
        help="Duration in seconds for each layer's sound (default: 0.15)"
    )
    parser.add_argument(
        "--min-freq", type=float, default=80.0,
        help="Minimum frequency for sonification (default: 80 Hz)"
    )
    parser.add_argument(
        "--max-freq", type=float, default=2000.0,
        help="Maximum frequency for sonification (default: 2000 Hz)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./sonification_output",
        help="Directory to save WAV files (default: ./sonification_output)"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port for the web interface (default: 5000)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    parser.add_argument(
        "--sonification-mode", type=str, default="spectral",
        choices=["spectral", "amplitude", "granular", "harmonic"],
        help="How to convert activations to sound (default: spectral)"
    )
    return parser.parse_args()


# ============================================================
# SONIFICATION ENGINE
# ============================================================

class LayerSonifier:
    """Converts neural network layer activations into audio signals."""

    def __init__(self, sample_rate=22050, duration_per_layer=0.15,
                 min_freq=80.0, max_freq=2000.0, mode="spectral"):
        self.sample_rate = sample_rate
        self.duration_per_layer = duration_per_layer
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.mode = mode

    def activations_to_audio(self, layer_activations, layer_index, total_layers):
        """
        Convert a single layer's activations to an audio signal.
        
        Args:
            layer_activations: numpy array of activation values
            layer_index: which layer this is (0-indexed)
            total_layers: total number of layers
        
        Returns:
            numpy array of audio samples
        """
        num_samples = int(self.sample_rate * self.duration_per_layer)
        t = np.linspace(0, self.duration_per_layer, num_samples, endpoint=False)

        # Normalize activations to [0, 1]
        act = layer_activations.flatten().astype(np.float64)
        act_min = act.min()
        act_max = act.max()
        if act_max - act_min > 1e-8:
            act_norm = (act - act_min) / (act_max - act_min)
        else:
            act_norm = np.zeros_like(act)

        if self.mode == "spectral":
            return self._spectral_mode(act_norm, t, layer_index, total_layers)
        elif self.mode == "amplitude":
            return self._amplitude_mode(act_norm, t, layer_index, total_layers)
        elif self.mode == "granular":
            return self._granular_mode(act_norm, t, layer_index, total_layers)
        elif self.mode == "harmonic":
            return self._harmonic_mode(act_norm, t, layer_index, total_layers)
        else:
            return self._spectral_mode(act_norm, t, layer_index, total_layers)

    def _spectral_mode(self, act_norm, t, layer_index, total_layers):
        """Map activation statistics to frequency components."""
        # Use statistical properties of activations to determine frequencies
        mean_val = np.mean(act_norm)
        std_val = np.std(act_norm)
        skew_val = np.mean((act_norm - mean_val) ** 3) / (std_val ** 3 + 1e-8)

        # Base frequency depends on layer position
        layer_ratio = layer_index / max(total_layers - 1, 1)
        base_freq = self.min_freq + layer_ratio * (self.max_freq - self.min_freq)

        # Modulate with activation statistics
        freq1 = base_freq * (1 + 0.5 * mean_val)
        freq2 = base_freq * (1 + std_val) * 1.5
        freq3 = base_freq * (2 + 0.3 * skew_val)

        # Generate composite waveform
        signal = (
            0.5 * np.sin(2 * np.pi * freq1 * t) +
            0.3 * np.sin(2 * np.pi * freq2 * t) * np.exp(-t * 3) +
            0.2 * np.sin(2 * np.pi * freq3 * t) * (1 - t / t[-1])
        )

        # Apply envelope
        envelope = np.exp(-t * 2) * (1 - np.exp(-t * 50))
        signal *= envelope

        return signal

    def _amplitude_mode(self, act_norm, t, layer_index, total_layers):
        """Use activations directly as amplitude modulation."""
        layer_ratio = layer_index / max(total_layers - 1, 1)
        carrier_freq = self.min_freq + layer_ratio * (self.max_freq - self.min_freq)

        # Resample activations to match audio length
        num_samples = len(t)
        indices = np.linspace(0, len(act_norm) - 1, num_samples).astype(int)
        amp_envelope = act_norm[indices]

        # Smooth the envelope
        kernel_size = max(1, num_samples // 50)
        kernel = np.ones(kernel_size) / kernel_size
        amp_envelope = np.convolve(amp_envelope, kernel, mode='same')

        signal = amp_envelope * np.sin(2 * np.pi * carrier_freq * t)
        envelope = np.exp(-t * 1.5) * (1 - np.exp(-t * 30))
        signal *= envelope

        return signal

    def _granular_mode(self, act_norm, t, layer_index, total_layers):
        """Create granular synthesis from activation patterns."""
        num_samples = len(t)
        signal = np.zeros(num_samples)

        # Create grains based on activation peaks
        num_grains = min(20, len(act_norm) // 10)
        grain_indices = np.argsort(act_norm)[-num_grains:]

        for i, idx in enumerate(grain_indices):
            grain_freq = self.min_freq + act_norm[idx] * (self.max_freq - self.min_freq)
            grain_pos = int((idx / len(act_norm)) * num_samples)
            grain_len = min(num_samples // num_grains, num_samples - grain_pos)

            if grain_len > 0:
                grain_t = np.linspace(0, grain_len / self.sample_rate, grain_len)
                grain = np.sin(2 * np.pi * grain_freq * grain_t)
                # Hann window
                window = np.hanning(grain_len)
                grain *= window * act_norm[idx]
                signal[grain_pos:grain_pos + grain_len] += grain

        # Normalize
        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal /= max_abs

        return signal

    def _harmonic_mode(self, act_norm, t, layer_index, total_layers):
        """Map top activations to harmonic series."""
        layer_ratio = layer_index / max(total_layers - 1, 1)
        fundamental = self.min_freq + layer_ratio * (self.max_freq - self.min_freq) * 0.3

        # Use top activation values as harmonic amplitudes
        num_harmonics = min(12, len(act_norm))
        top_vals = np.sort(act_norm)[-num_harmonics:]

        signal = np.zeros_like(t)
        for h, amp in enumerate(top_vals, 1):
            freq = fundamental * h
            if freq < self.sample_rate / 2:  # Nyquist
                signal += amp * np.sin(2 * np.pi * freq * t) / h

        envelope = np.exp(-t * 2) * (1 - np.exp(-t * 40))
        signal *= envelope

        max_abs = np.max(np.abs(signal))
        if max_abs > 0:
            signal /= max_abs

        return signal

    def generate_step_audio(self, all_layer_activations):
        """
        Generate audio for one complete generation step (all layers).
        
        Args:
            all_layer_activations: list of numpy arrays, one per layer
        
        Returns:
            numpy array of concatenated audio for all layers
        """
        total_layers = len(all_layer_activations)
        audio_segments = []

        for i, layer_act in enumerate(all_layer_activations):
            segment = self.activations_to_audio(layer_act, i, total_layers)
            audio_segments.append(segment)

        # Concatenate all layer sounds with tiny crossfade
        full_audio = np.concatenate(audio_segments)

        # Normalize final output
        max_abs = np.max(np.abs(full_audio))
        if max_abs > 0:
            full_audio = full_audio / max_abs * 0.9

        return full_audio

    def save_wav(self, audio_data, filepath):
        """Save audio data as a WAV file."""
        # Convert to 16-bit PCM
        audio_16bit = (audio_data * 32767).astype(np.int16)

        with wave.open(str(filepath), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())


# ============================================================
# LLM WITH LAYER EXTRACTION
# ============================================================

class LLMSonificationEngine:
    """Loads an LLM and extracts layer activations during generation."""

    def __init__(self, model_name="gpt2", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model '{model_name}' on {self.device}...")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name, output_hidden_states=True
        ).to(self.device)
        self.model.eval()

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.num_layers = self.model.config.n_layer + 1  # +1 for embedding layer
        print(f"Model loaded! {self.num_layers} layers, "
              f"{self.model.config.n_embd} hidden dim")

    def generate_step(self, input_ids, temperature=0.8, top_k=50):
        """
        Run one forward pass and return:
        - next token id
        - layer activations (list of numpy arrays)
        - top-5 predicted tokens with probabilities
        """
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)

        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors

        # Extract activations for the last token position
        layer_activations = []
        for hs in hidden_states:
            # Shape: (batch, seq_len, hidden_dim) -> take last token
            last_token_act = hs[0, -1, :].cpu().numpy()
            # Subsample for sonification (take every Nth value)
            subsample = last_token_act[::4]  # Reduce dimensionality
            layer_activations.append(subsample)

        # Get logits for next token prediction
        logits = outputs.logits[0, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(0, top_k_indices, top_k_logits)
        else:
            logits_filtered = logits

        probs = torch.softmax(logits_filtered, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_tokens = [
            {
                "token": self.tokenizer.decode([idx.item()]),
                "probability": prob.item()
            }
            for prob, idx in zip(top5_probs, top5_indices)
        ]

        return next_token.item(), layer_activations, top5_tokens

    def generate_full(self, prompt, max_tokens=50, temperature=0.8, top_k=50,
                      sonifier=None, output_dir=None, callback=None):
        """
        Generate tokens one by one, extracting layer activations at each step.
        
        Args:
            prompt: input text
            max_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            sonifier: LayerSonifier instance
            output_dir: directory to save WAV files
            callback: function called after each step with results
        
        Returns:
            dict with generated text, audio files, and step data
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated_tokens = []
        steps_data = []
        all_audio = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for step in range(max_tokens):
            next_token, layer_activations, top5 = self.generate_step(
                input_ids, temperature, top_k
            )

            token_text = self.tokenizer.decode([next_token])
            generated_tokens.append(next_token)

            # Generate audio for this step
            step_audio = None
            wav_path = None
            if sonifier:
                step_audio = sonifier.generate_step_audio(layer_activations)
                all_audio.append(step_audio)

                if output_dir:
                    wav_path = os.path.join(output_dir, f"step_{step:04d}.wav")
                    sonifier.save_wav(step_audio, wav_path)

            # Compute layer statistics for visualization
            layer_stats = []
            for i, act in enumerate(layer_activations):
                layer_stats.append({
                    "layer": i,
                    "mean": float(np.mean(act)),
                    "std": float(np.std(act)),
                    "min": float(np.min(act)),
                    "max": float(np.max(act)),
                    "energy": float(np.sum(act ** 2)),
                })

            step_info = {
                "step": step,
                "token": token_text,
                "token_id": next_token,
                "top5": top5,
                "layer_stats": layer_stats,
                "wav_path": wav_path,
            }
            steps_data.append(step_info)

            if callback:
                callback(step_info)

            # Append token to input
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)

            # Stop at EOS
            if next_token == self.tokenizer.eos_token_id:
                break

        # Save combined audio
        combined_wav_path = None
        if sonifier and all_audio and output_dir:
            combined_audio = np.concatenate(all_audio)
            combined_wav_path = os.path.join(output_dir, "full_generation.wav")
            sonifier.save_wav(combined_audio, combined_wav_path)

        generated_text = self.tokenizer.decode(generated_tokens)

        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "full_text": prompt + generated_text,
            "steps": steps_data,
            "combined_wav": combined_wav_path,
            "num_steps": len(steps_data),
        }


# ============================================================
# WEB INTERFACE (Flask)
# ============================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🧠 LLM Layer Sonification</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a1a;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        
        .controls {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }
        .controls h2 { color: #667eea; margin-bottom: 15px; }
        .control-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .control-group { display: flex; flex-direction: column; }
        .control-group label {
            font-size: 0.85em;
            color: #aaa;
            margin-bottom: 5px;
        }
        .control-group input, .control-group select, .control-group textarea {
            background: #0f0f23;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 10px;
            color: #fff;
            font-size: 0.95em;
        }
        .control-group textarea { resize: vertical; min-height: 60px; }
        .control-group input:focus, .control-group select:focus {
            border-color: #667eea;
            outline: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 8px;
            padding: 12px 30px;
            color: white;
            font-size: 1.1em;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 20px rgba(102,126,234,0.4); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .output-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }
        @media (max-width: 900px) { .output-section { grid-template-columns: 1fr; } }
        
        .panel {
            background: #1a1a2e;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #333;
        }
        .panel h3 { color: #764ba2; margin-bottom: 15px; }
        
        .generated-text {
            font-family: 'Courier New', monospace;
            background: #0f0f23;
            padding: 15px;
            border-radius: 8px;
            line-height: 1.6;
            max-height: 300px;
            overflow-y: auto;
        }
        .generated-text .prompt { color: #667eea; }
        .generated-text .generated { color: #4ecdc4; }
        .generated-text .current { color: #ff6b6b; font-weight: bold; }
        
        .step-info {
            max-height: 400px;
            overflow-y: auto;
        }
        .step-card {
            background: #0f0f23;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 3px solid #667eea;
        }
        .step-card .token { color: #4ecdc4; font-weight: bold; font-size: 1.2em; }
        .step-card .predictions { font-size: 0.85em; color: #aaa; margin-top: 5px; }
        
        .audio-player {
            width: 100%;
            margin: 10px 0;
        }
        
        .layer-viz {
            display: flex;
            gap: 2px;
            align-items: flex-end;
            height: 80px;
            margin: 10px 0;
            padding: 5px;
            background: #0f0f23;
            border-radius: 6px;
        }
        .layer-bar {
            flex: 1;
            background: linear-gradient(to top, #667eea, #764ba2);
            border-radius: 2px 2px 0 0;
            min-width: 3px;
            transition: height 0.3s;
        }
        
        .status {
            text-align: center;
            padding: 10px;
            color: #888;
            font-style: italic;
        }
        .status.running { color: #4ecdc4; }
        .status.done { color: #667eea; }
        
        .full-audio { margin-top: 20px; }
        
        #progress-bar {
            width: 100%;
            height: 6px;
            background: #333;
            border-radius: 3px;
            margin: 15px 0;
            overflow: hidden;
        }
        #progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 LLM Layer Sonification</h1>
        <p class="subtitle">Hear the neural network think — every layer, every token</p>
        
        <div class="controls">
            <h2>⚙️ Controls</h2>
            <div class="control-grid">
                <div class="control-group">
                    <label>Prompt</label>
                    <textarea id="prompt">{{ default_prompt }}</textarea>
                </div>
                <div class="control-group">
                    <label>Model</label>
                    <select id="model">
                        <option value="gpt2" {{ 'selected' if model_name == 'gpt2' }}>GPT-2 (124M)</option>
                        <option value="distilgpt2" {{ 'selected' if model_name == 'distilgpt2' }}>DistilGPT-2 (82M)</option>
                        <option value="gpt2-medium" {{ 'selected' if model_name == 'gpt2-medium' }}>GPT-2 Medium (355M)</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Max Tokens: <span id="max-tokens-val">{{ max_tokens }}</span></label>
                    <input type="range" id="max-tokens" min="5" max="200" value="{{ max_tokens }}"
                           oninput="document.getElementById('max-tokens-val').textContent=this.value">
                </div>
                <div class="control-group">
                    <label>Temperature: <span id="temp-val">{{ temperature }}</span></label>
                    <input type="range" id="temperature" min="0.1" max="2.0" step="0.1" value="{{ temperature }}"
                           oninput="document.getElementById('temp-val').textContent=this.value">
                </div>
                <div class="control-group">
                    <label>Top-K: <span id="topk-val">{{ top_k }}</span></label>
                    <input type="range" id="top-k" min="1" max="100" value="{{ top_k }}"
                           oninput="document.getElementById('topk-val').textContent=this.value">
                </div>
                <div class="control-group">
                    <label>Sonification Mode</label>
                    <select id="sonification-mode">
                        <option value="spectral" {{ 'selected' if son_mode == 'spectral' }}>Spectral</option>
                        <option value="amplitude" {{ 'selected' if son_mode == 'amplitude' }}>Amplitude</option>
                        <option value="granular" {{ 'selected' if son_mode == 'granular' }}>Granular</option>
                        <option value="harmonic" {{ 'selected' if son_mode == 'harmonic' }}>Harmonic</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Min Freq (Hz): <span id="minfreq-val">{{ min_freq }}</span></label>
                    <input type="range" id="min-freq" min="20" max="500" value="{{ min_freq }}"
                           oninput="document.getElementById('minfreq-val').textContent=this.value">
                </div>
                <div class="control-group">
                    <label>Max Freq (Hz): <span id="maxfreq-val">{{ max_freq }}</span></label>
                    <input type="range" id="max-freq" min="500" max="8000" value="{{ max_freq }}"
                           oninput="document.getElementById('maxfreq-val').textContent=this.value">
                </div>
                <div class="control-group">
                    <label>Duration/Layer (s): <span id="dur-val">{{ duration }}</span></label>
                    <input type="range" id="duration" min="0.05" max="0.5" step="0.05" value="{{ duration }}"
                           oninput="document.getElementById('dur-val').textContent=this.value">
                </div>
            </div>
            <button class="btn" id="generate-btn" onclick="startGeneration()">
                🎵 Generate & Sonify
            </button>
        </div>
        
        <div id="progress-bar" style="display:none;">
            <div id="progress-fill"></div>
        </div>
        <p class="status" id="status">Ready to generate</p>
        
        <div class="output-section">
            <div class="panel">
                <h3>📝 Generated Text</h3>
                <div class="generated-text" id="text-output">
                    <span class="prompt">Waiting for generation...</span>
                </div>
                <div class="full-audio" id="full-audio-container" style="display:none;">
                    <h3>🔊 Full Generation Audio</h3>
                    <audio id="full-audio" class="audio-player" controls></audio>
                </div>
            </div>
            
            <div class="panel">
                <h3>🔬 Step Details</h3>
                <div class="step-info" id="step-info">
                    <p style="color:#888;">Steps will appear here during generation...</p>
                </div>
            </div>
        </div>
        
        <div class="panel" style="margin-top:20px;">
            <h3>📊 Layer Activations (Current Step)</h3>
            <div class="layer-viz" id="layer-viz">
                <!-- Bars will be added dynamically -->
            </div>
            <p style="font-size:0.8em; color:#666; margin-top:5px;">
                Each bar = one layer's energy. Height = activation magnitude.
            </p>
        </div>
    </div>
    
    <script>
        let generating = false;
        let eventSource = null;
        
        function startGeneration() {
            if (generating) return;
            generating = true;
            document.getElementById('generate-btn').disabled = true;
            document.getElementById('status').textContent = 'Generating...';
            document.getElementById('status').className = 'status running';
            document.getElementById('progress-bar').style.display = 'block';
            document.getElementById('progress-fill').style.width = '0%';
            document.getElementById('step-info').innerHTML = '';
            document.getElementById('full-audio-container').style.display = 'none';
            
            const params = {
                prompt: document.getElementById('prompt').value,
                model: document.getElementById('model').value,
                max_tokens: parseInt(document.getElementById('max-tokens').value),
                temperature: parseFloat(document.getElementById('temperature').value),
                top_k: parseInt(document.getElementById('top-k').value),
                sonification_mode: document.getElementById('sonification-mode').value,
                min_freq: parseFloat(document.getElementById('min-freq').value),
                max_freq: parseFloat(document.getElementById('max-freq').value),
                duration_per_layer: parseFloat(document.getElementById('duration').value),
            };
            
            // Update text display with prompt
            document.getElementById('text-output').innerHTML = 
                `<span class="prompt">${escapeHtml(params.prompt)}</span><span class="generated" id="gen-text"></span>`;
            
            // Start SSE connection for streaming results
            const queryString = new URLSearchParams(params).toString();
            eventSource = new EventSource(`/generate?${queryString}`);
            
            let stepCount = 0;
            
            eventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'step') {
                    stepCount++;
                    const progress = (stepCount / params.max_tokens) * 100;
                    document.getElementById('progress-fill').style.width = `${progress}%`;
                    
                    // Update generated text
                    const genTextEl = document.getElementById('gen-text');
                    genTextEl.textContent += data.token;
                    
                    // Add step card
                    const stepHtml = createStepCard(data);
                    document.getElementById('step-info').innerHTML = stepHtml + 
                        document.getElementById('step-info').innerHTML;
                    
                    // Update layer visualization
                    updateLayerViz(data.layer_stats);
                    
                    // Play step audio if available
                    if (data.audio_b64) {
                        playAudioBase64(data.audio_b64);
                    }
                    
                } else if (data.type === 'done') {
                    generating = false;
                    document.getElementById('generate-btn').disabled = false;
                    document.getElementById('status').textContent = 
                        `Done! Generated ${data.num_steps} tokens.`;
                    document.getElementById('status').className = 'status done';
                    document.getElementById('progress-fill').style.width = '100%';
                    
                    // Show full audio player
                    if (data.full_audio_url) {
                        document.getElementById('full-audio-container').style.display = 'block';
                        document.getElementById('full-audio').src = data.full_audio_url;
                    }
                    
                    eventSource.close();
                    eventSource = null;
                    
                } else if (data.type === 'error') {
                    generating = false;
                    document.getElementById('generate-btn').disabled = false;
                    document.getElementById('status').textContent = `Error: ${data.message}`;
                    document.getElementById('status').className = 'status';
                    eventSource.close();
                    eventSource = null;
                }
            };
            
            eventSource.onerror = function() {
                generating = false;
                document.getElementById('generate-btn').disabled = false;
                document.getElementById('status').textContent = 'Connection lost.';
                document.getElementById('status').className = 'status';
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                }
            };
        }
        
        function createStepCard(data) {
            const top5Html = data.top5.map(t => 
                `<span style="color:#667eea;">"${escapeHtml(t.token)}"</span> (${(t.probability*100).toFixed(1)}%)`
            ).join(', ');
            
            return `
                <div class="step-card">
                    <span class="token">Step ${data.step}: "${escapeHtml(data.token)}"</span>
                    <div class="predictions">Top 5: ${top5Html}</div>
                    ${data.audio_b64 ? `<audio class="audio-player" controls src="data:audio/wav;base64,${data.audio_b64}" style="height:30px;width:100%;margin-top:5px;"></audio>` : ''}
                </div>
            `;
        }
        
        function updateLayerViz(layerStats) {
            const container = document.getElementById('layer-viz');
            if (!layerStats || layerStats.length === 0) return;
            
            // Normalize energies for display
            const energies = layerStats.map(s => s.energy);
            const maxEnergy = Math.max(...energies);
            
            let barsHtml = '';
            for (let i = 0; i < layerStats.length; i++) {
                const height = maxEnergy > 0 ? (energies[i] / maxEnergy) * 100 : 10;
                const hue = (i / layerStats.length) * 120 + 240; // blue to purple
                barsHtml += `<div class="layer-bar" style="height:${Math.max(height, 2)}%; background:hsl(${hue},70%,60%);" title="Layer ${i}: energy=${energies[i].toFixed(2)}"></div>`;
            }
            container.innerHTML = barsHtml;
        }
        
        function playAudioBase64(b64Data) {
            try {
                const audio = new Audio(`data:audio/wav;base64,${b64Data}`);
                audio.volume = 0.5;
                audio.play().catch(e => console.log('Autoplay blocked:', e));
            } catch(e) {
                console.log('Audio play error:', e);
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# ============================================================
# FLASK APP
# ============================================================

def create_app(args):
    app = Flask(__name__)
    app.config['args'] = args

    # Initialize engine (will be loaded on first request or at startup)
    engine = None
    engine_lock = threading.Lock()

    def get_engine(model_name=None):
        nonlocal engine
        target_model = model_name or args.model
        with engine_lock:
            if engine is None or engine.model_name != target_model:
                engine = LLMSonificationEngine(target_model)
        return engine

    # Pre-load model at startup
    get_engine()

    @app.route('/')
    def index():
        return render_template_string(
            HTML_TEMPLATE,
            default_prompt=args.prompt,
            model_name=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            son_mode=args.sonification_mode,
            min_freq=int(args.min_freq),
            max_freq=int(args.max_freq),
            duration=args.duration_per_layer,
        )

    @app.route('/generate')
    def generate():
        """Server-Sent Events endpoint for streaming generation."""
        prompt = request.args.get('prompt', args.prompt)
        model_name = request.args.get('model', args.model)
        max_tokens = int(request.args.get('max_tokens', args.max_tokens))
        temperature = float(request.args.get('temperature', args.temperature))
        top_k = int(request.args.get('top_k', args.top_k))
        son_mode = request.args.get('sonification_mode', args.sonification_mode)
        min_freq = float(request.args.get('min_freq', args.min_freq))
        max_freq = float(request.args.get('max_freq', args.max_freq))
        duration_per_layer = float(request.args.get('duration_per_layer', args.duration_per_layer))

        def event_stream():
            try:
                eng = get_engine(model_name)
                sonifier = LayerSonifier(
                    sample_rate=args.sample_rate,
                    duration_per_layer=duration_per_layer,
                    min_freq=min_freq,
                    max_freq=max_freq,
                    mode=son_mode,
                )

                output_dir = args.output_dir
                os.makedirs(output_dir, exist_ok=True)

                input_ids = eng.tokenizer.encode(prompt, return_tensors="pt").to(eng.device)
                all_audio = []

                for step in range(max_tokens):
                    next_token, layer_activations, top5 = eng.generate_step(
                        input_ids, temperature, top_k
                    )

                    token_text = eng.tokenizer.decode([next_token])

                    # Generate audio for this step
                    step_audio = sonifier.generate_step_audio(layer_activations)
                    all_audio.append(step_audio)

                    # Save step WAV
                    wav_path = os.path.join(output_dir, f"step_{step:04d}.wav")
                    sonifier.save_wav(step_audio, wav_path)

                    # Encode audio as base64 for browser playback
                    audio_16bit = (step_audio * 32767).astype(np.int16)
                    import io
                    buf = io.BytesIO()
                    with wave.open(buf, 'w') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(args.sample_rate)
                        wf.writeframes(audio_16bit.tobytes())
                    audio_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

                    # Layer statistics
                    layer_stats = []
                    for i, act in enumerate(layer_activations):
                        layer_stats.append({
                            "layer": i,
                            "mean": float(np.mean(act)),
                            "std": float(np.std(act)),
                            "min": float(np.min(act)),
                            "max": float(np.max(act)),
                            "energy": float(np.sum(act ** 2)),
                        })

                    step_data = {
                        "type": "step",
                        "step": step,
                        "token": token_text,
                        "token_id": next_token,
                        "top5": top5,
                        "layer_stats": layer_stats,
                        "audio_b64": audio_b64,
                    }

                    yield f"data: {json.dumps(step_data)}\n\n"

                    # Update input_ids
                    input_ids = torch.cat([
                        input_ids,
                        torch.tensor([[next_token]], device=eng.device)
                    ], dim=1)

                    # Stop at EOS
                    if next_token == eng.tokenizer.eos_token_id:
                        break

                # Save combined audio
                combined_audio = np.concatenate(all_audio)
                combined_path = os.path.join(output_dir, "full_generation.wav")
                sonifier.save_wav(combined_audio, combined_path)

                done_data = {
                    "type": "done",
                    "num_steps": step + 1,
                    "full_audio_url": "/audio/full_generation.wav",
                }
                yield f"data: {json.dumps(done_data)}\n\n"

            except Exception as e:
                error_data = {"type": "error", "message": str(e)}
                yield f"data: {json.dumps(error_data)}\n\n"

        from flask import Response
        return Response(
            event_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
            }
        )

    @app.route('/audio/<filename>')
    def serve_audio(filename):
        """Serve generated WAV files."""
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            return send_file(filepath, mimetype='audio/wav')
        return "Not found", 404

    return app


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("  🧠 LLM Layer Sonification")
    print("  Hear the neural network think!")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Prompt:         {args.prompt[:50]}...")
    print(f"  Max tokens:     {args.max_tokens}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Top-K:          {args.top_k}")
    print(f"  Sample rate:    {args.sample_rate} Hz")
    print(f"  Duration/layer: {args.duration_per_layer}s")
    print(f"  Freq range:     {args.min_freq}-{args.max_freq} Hz")
    print(f"  Mode:           {args.sonification_mode}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Web interface:  http://localhost:{args.port}")
    print("=" * 60)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create and run Flask app
    app = create_app(args)
    print(f"\n🌐 Open your browser to: http://localhost:{args.port}\n")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
