"""
grokking_transformer_sim.py

A minimal, fully pre-initialized simulation of the one-layer transformer
algorithm for modular addition (mod P), as reverse-engineered in:

  "Progress Measures for Grokking via Mechanistic Interpretability"
  Nanda et al., ICLR 2023

The learned algorithm:
  1. Embed inputs a, b into sin/cos at sparse "key frequencies"
  2. Use attention + MLP to compute sin(wk*(a+b)) and cos(wk*(a+b))
     via trigonometric identities
  3. Unembed by computing cos(wk*(a+b-c)) for each candidate output c
  4. Sum over frequencies → constructive interference at c* = (a+b) mod P

Every function is isolated so you can set breakpoints and inspect shapes,
values, and intermediate activations.

Usage:
    python grokking_transformer_sim.py
    # or step through with: python -m pdb grokking_transformer_sim.py
"""

import numpy as np

# =============================================================================
# HYPERPARAMETERS (matching the paper's mainline model)
# =============================================================================
P = 113                          # Prime modulus
KEY_FREQS = [14, 35, 41, 42, 52] # The 5 key frequencies discovered in the model
D_MODEL = 128                    # Embedding dimension
D_HEAD = 32                      # Attention head dimension
N_HEADS = 4                      # Number of attention heads
D_MLP = 512                      # MLP hidden dimension (number of neurons)
N_KEY = len(KEY_FREQS)           # 5 key frequencies

# Derived: angular frequencies w_k = 2*pi*k / P
W_KEY = np.array([2.0 * np.pi * k / P for k in KEY_FREQS])  # shape: (5,)

# =============================================================================
# UTILITY: SOFTMAX
# =============================================================================
def softmax(x):
    """
    Standard softmax over the last axis.
    Numerically stable version.
    
    Args:
        x: numpy array of any shape
    Returns:
        softmax probabilities, same shape as x
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# =============================================================================
# UTILITY: RELU
# =============================================================================
def relu(x):
    """
    Rectified Linear Unit activation.
    
    Args:
        x: numpy array of any shape
    Returns:
        max(0, x) element-wise
    """
    return np.maximum(0.0, x)

# =============================================================================
# UTILITY: SIGMOID (used for 2-element softmax)
# =============================================================================
def sigmoid(x):
    """
    Logistic sigmoid function.
    When softmax is over 2 elements, it reduces to sigmoid of their difference.
    
    Args:
        x: scalar or numpy array
    Returns:
        1 / (1 + exp(-x))
    """
    return 1.0 / (1.0 + np.exp(-x))

# =============================================================================
# DENSE (Linear layer, no bias)
# =============================================================================
def dense(x, W):
    """
    A simple linear (dense/fully-connected) layer: y = x @ W^T
    
    Args:
        x: input vector/matrix, shape (..., in_features)
        W: weight matrix, shape (out_features, in_features)
    Returns:
        output, shape (..., out_features)
    """
    return x @ W.T

# =============================================================================
# DENSE WITH BIAS
# =============================================================================
def dense_bias(x, W, b):
    """
    Linear layer with bias: y = x @ W^T + b
    
    Args:
        x: input, shape (..., in_features)
        W: weight matrix, shape (out_features, in_features)
        b: bias vector, shape (out_features,)
    Returns:
        output, shape (..., out_features)
    """
    return x @ W.T + b

# =============================================================================
# EMBEDDING: Map one-hot token to sin/cos representation
# =============================================================================
def embed(token_id, W_E):
    """
    Look up the embedding for a token (equivalent to W_E @ one_hot).
    In the learned model, this produces vectors rich in sin(wk*a), cos(wk*a).
    
    Args:
        token_id: integer in [0, P-1]
        W_E: embedding matrix, shape (D_MODEL, P)
    Returns:
        embedding vector, shape (D_MODEL,)
    """
    # One-hot lookup is just selecting a column
    return W_E[:, token_id]

# =============================================================================
# POSITIONAL EMBEDDING
# =============================================================================
def add_positional(embedding, pos, W_pos):
    """
    Add learned positional embedding.
    
    Args:
        embedding: token embedding, shape (D_MODEL,)
        pos: position index (0 for 'a', 1 for 'b', 2 for '=')
        W_pos: positional embedding matrix, shape (3, D_MODEL)
    Returns:
        embedding + positional, shape (D_MODEL,)
    """
    return embedding + W_pos[pos]

# =============================================================================
# ATTENTION SCORE (single head, from '=' to token at position i)
# =============================================================================
def attention_score(x_query, x_key, W_Q, W_K):
    """
    Compute raw attention score: q^T k / sqrt(d_head)
    
    In the grokking model, the attention from '=' to 'a' is approximately:
        0.5 + alpha*(cos(wk*a) - cos(wk*b)) + beta*(sin(wk*a) - sin(wk*b))
    i.e., periodic in a single key frequency.
    
    Args:
        x_query: query token residual stream, shape (D_MODEL,)
        x_key: key token residual stream, shape (D_MODEL,)
        W_Q: query weight, shape (D_HEAD, D_MODEL)
        W_K: key weight, shape (D_HEAD, D_MODEL)
    Returns:
        scalar attention score
    """
    q = W_Q @ x_query   # shape (D_HEAD,)
    k = W_K @ x_key     # shape (D_HEAD,)
    score = np.dot(q, k) / np.sqrt(D_HEAD)
    return score

# =============================================================================
# ATTENTION PATTERN (simplified: softmax over 2 positions = sigmoid)
# =============================================================================
def attention_pattern(score_a, score_b):
    """
    Compute attention weights from '=' to positions 'a' and 'b'.
    Since attention to '=' itself is negligible, this is sigmoid of difference.
    
    Args:
        score_a: raw score for position 0 (token a)
        score_b: raw score for position 1 (token b)
    Returns:
        (weight_a, weight_b) tuple, summing to ~1
    """
    # Softmax over 2 elements = sigmoid of difference
    weight_a = sigmoid(score_a - score_b)
    weight_b = 1.0 - weight_a
    return weight_a, weight_b

# =============================================================================
# ATTENTION HEAD OUTPUT (OV circuit)
# =============================================================================
def attention_head(x_a, x_b, weight_a, weight_b, W_V, W_O):
    """
    Compute the output of one attention head.
    
    output = W_O @ (weight_a * W_V @ x_a + weight_b * W_V @ x_b)
    
    In the grokking model, this produces degree-2 polynomials of sin/cos
    at a single key frequency (for heads 0 and 2), or amplifies embeddings
    (for heads 1 and 3).
    
    Args:
        x_a: residual stream at position 0, shape (D_MODEL,)
        x_b: residual stream at position 1, shape (D_MODEL,)
        weight_a: attention weight to position a (scalar)
        weight_b: attention weight to position b (scalar)
        W_V: value weight, shape (D_HEAD, D_MODEL)
        W_O: output weight, shape (D_MODEL, D_HEAD)
    Returns:
        head output, shape (D_MODEL,)
    """
    v_a = W_V @ x_a  # shape (D_HEAD,)
    v_b = W_V @ x_b  # shape (D_HEAD,)
    # Weighted combination
    v_combined = weight_a * v_a + weight_b * v_b  # shape (D_HEAD,)
    # Project back to residual stream
    output = W_O @ v_combined  # shape (D_MODEL,)
    return output

# =============================================================================
# MULTI-HEAD ATTENTION (all heads summed)
# =============================================================================
def multi_head_attention(x_a, x_b, x_eq, heads_params):
    """
    Full multi-head attention layer output on the '=' token.
    
    Args:
        x_a: residual stream at position 0 (token a), shape (D_MODEL,)
        x_b: residual stream at position 1 (token b), shape (D_MODEL,)
        x_eq: residual stream at position 2 (token '='), shape (D_MODEL,)
        heads_params: list of dicts, each with keys 'W_Q', 'W_K', 'W_V', 'W_O'
    Returns:
        attention output added to residual stream of '=', shape (D_MODEL,)
    """
    attn_output = np.zeros(D_MODEL)
    
    for head in heads_params:
        # Compute scores from '=' to 'a' and 'b'
        score_a = attention_score(x_eq, x_a, head['W_Q'], head['W_K'])
        score_b = attention_score(x_eq, x_b, head['W_Q'], head['W_K'])
        
        # Get attention weights
        w_a, w_b = attention_pattern(score_a, score_b)
        
        # Compute head output
        h_out = attention_head(x_a, x_b, w_a, w_b, head['W_V'], head['W_O'])
        attn_output += h_out
    
    # Residual connection
    return x_eq + attn_output

# =============================================================================
# MLP LAYER (ReLU activation)
# =============================================================================
def mlp(x, W_in, b_in, W_out, b_out):
    """
    One-hidden-layer MLP with ReLU:
        hidden = ReLU(W_in @ x + b_in)
        output = W_out @ hidden + b_out
    
    In the grokking model, ~85% of neurons compute degree-2 polynomials
    of sin/cos at a single key frequency. After ReLU, these encode
    cos(wk*(a+b)) and sin(wk*(a+b)) via trig identities.
    
    Args:
        x: input from residual stream, shape (D_MODEL,)
        W_in: input weights, shape (D_MLP, D_MODEL)
        b_in: input bias, shape (D_MLP,)
        W_out: output weights, shape (D_MODEL, D_MLP)
        b_out: output bias, shape (D_MODEL,)
    Returns:
        mlp_output: shape (D_MODEL,)
        hidden_activations: shape (D_MLP,) — for inspection
    """
    # Pre-activation
    pre_act = W_in @ x + b_in          # shape (D_MLP,)
    
    # ReLU activation — this is where trig identities get computed!
    # cos(wk*a)*cos(wk*b) - sin(wk*a)*sin(wk*b) = cos(wk*(a+b))
    hidden = relu(pre_act)              # shape (D_MLP,)
    
    # Project back
    output = W_out @ hidden + b_out     # shape (D_MODEL,)
    
    return output, hidden

# =============================================================================
# UNEMBEDDING: Compute logits for all candidate outputs c ∈ {0, ..., P-1}
# =============================================================================
def unembed(x, W_U):
    """
    Compute logits: logits = W_U @ x
    
    In the grokking model, W_U encodes cos(wk*c) and sin(wk*c) so that
    the dot product computes cos(wk*(a+b-c)) via:
        cos(wk*(a+b))*cos(wk*c) + sin(wk*(a+b))*sin(wk*c)
    
    Constructive interference at c* = (a+b) mod P gives the largest logit.
    
    Args:
        x: final residual stream, shape (D_MODEL,)
        W_U: unembedding matrix, shape (P, D_MODEL)
    Returns:
        logits: shape (P,) — one score per candidate output
    """
    logits = W_U @ x  # shape (P,)
    return logits

# =============================================================================
# THE IDEALIZED FOURIER MULTIPLICATION ALGORITHM (analytical version)
# =============================================================================
def fourier_embed(token_id, freqs):
    """
    Idealized embedding: map token to sin/cos at each key frequency.
    
    This is what W_E effectively computes (sparse in Fourier basis).
    
    Args:
        token_id: integer in [0, P-1]
        freqs: array of angular frequencies, shape (N_KEY,)
    Returns:
        sin_components: shape (N_KEY,)
        cos_components: shape (N_KEY,)
    """
    angles = freqs * token_id  # shape (N_KEY,)
    return np.sin(angles), np.cos(angles)

def trig_combine(sin_a, cos_a, sin_b, cos_b):
    """
    Apply trigonometric addition identities:
        cos(wk*(a+b)) = cos(wk*a)*cos(wk*b) - sin(wk*a)*sin(wk*b)
        sin(wk*(a+b)) = sin(wk*a)*cos(wk*b) + cos(wk*a)*sin(wk*b)
    
    This is what the attention + MLP layers compute.
    
    Args:
        sin_a, cos_a: sin/cos of (wk * a), each shape (N_KEY,)
        sin_b, cos_b: sin/cos of (wk * b), each shape (N_KEY,)
    Returns:
        sin_ab: sin(wk*(a+b)), shape (N_KEY,)
        cos_ab: cos(wk*(a+b)), shape (N_KEY,)
    """
    cos_ab = cos_a * cos_b - sin_a * sin_b
    sin_ab = sin_a * cos_b + cos_a * sin_b
    return sin_ab, cos_ab

def compute_logits_fourier(sin_ab, cos_ab, freqs, alphas):
    """
    Compute logits for each candidate c using:
        logit(c) = sum_k alpha_k * cos(wk*(a+b-c))
                 = sum_k alpha_k * [cos(wk*(a+b))*cos(wk*c) + sin(wk*(a+b))*sin(wk*c)]
    
    Constructive interference: all cosines = 1 when c = (a+b) mod P.
    Destructive interference: cosines cancel out for other c values.
    
    Args:
        sin_ab: sin(wk*(a+b)), shape (N_KEY,)
        cos_ab: cos(wk*(a+b)), shape (N_KEY,)
        freqs: angular frequencies, shape (N_KEY,)
        alphas: amplitude weights per frequency, shape (N_KEY,)
    Returns:
        logits: shape (P,) — one per candidate output c
    """
    logits = np.zeros(P)
    
    for c in range(P):
        # For each candidate output, compute cos(wk*(a+b-c))
        for i, (wk, alpha) in enumerate(zip(freqs, alphas)):
            # cos(wk*(a+b-c)) = cos(wk*(a+b))*cos(wk*c) + sin(wk*(a+b))*sin(wk*c)
            cos_c = np.cos(wk * c)
            sin_c = np.sin(wk * c)
            logits[c] += alpha * (cos_ab[i] * cos_c + sin_ab[i] * sin_c)
    
    return logits

# =============================================================================
# FULL FORWARD PASS: IDEALIZED (analytical Fourier multiplication)
# =============================================================================
def forward_idealized(a, b, freqs=W_KEY, alphas=None):
    """
    Full forward pass of the idealized Fourier multiplication algorithm.
    
    This is the *analytical* version of what the trained transformer computes.
    You can step through each stage to see exactly how modular addition works.
    
    Args:
        a: first input, integer in [0, P-1]
        b: second input, integer in [0, P-1]
        freqs: key angular frequencies (default: from the paper)
        alphas: per-frequency amplitudes (default: equal weights)
    Returns:
        predicted_c: the model's prediction for (a + b) mod P
        logits: raw logit scores, shape (P,)
        intermediates: dict of all intermediate values for inspection
    """
    if alphas is None:
        # In the real model, these vary (see Table 1 in the paper: ~44 to ~68)
        alphas = np.array([44.1, 42.2, 44.8, 66.6, 63.0])
    
    intermediates = {}
    
    # --- Step 1: Embed inputs into sin/cos ---
    sin_a, cos_a = fourier_embed(a, freqs)
    sin_b, cos_b = fourier_embed(b, freqs)
    intermediates['sin_a'] = sin_a
    intermediates['cos_a'] = cos_a
    intermediates['sin_b'] = sin_b
    intermediates['cos_b'] = cos_b
    
    # --- Step 2: Trig identities (attention + MLP) ---
    sin_ab, cos_ab = trig_combine(sin_a, cos_a, sin_b, cos_b)
    intermediates['sin_ab'] = sin_ab  # sin(wk*(a+b))
    intermediates['cos_ab'] = cos_ab  # cos(wk*(a+b))
    
    # --- Step 3: Compute logits via unembedding ---
    logits = compute_logits_fourier(sin_ab, cos_ab, freqs, alphas)
    intermediates['logits'] = logits
    
    # --- Step 4: Predict output ---
    predicted_c = np.argmax(logits)
    intermediates['predicted_c'] = predicted_c
    intermediates['true_c'] = (a + b) % P
    intermediates['correct'] = (predicted_c == (a + b) % P)
    
    # --- Step 5: Softmax probabilities ---
    probs = softmax(logits)
    intermediates['probs'] = probs
    intermediates['prob_true'] = probs[(a + b) % P]
    
    return predicted_c, logits, intermediates

# =============================================================================
# FULL FORWARD PASS: SIMULATED TRANSFORMER (with actual weight matrices)
# =============================================================================
def initialize_weights(seed=42):
    """
    Initialize weight matrices that approximate the learned Fourier structure.
    
    We construct W_E to be sparse in the Fourier basis (only key frequencies),
    and W_U to read off cos(wk*c) and sin(wk*c).
    
    This gives you actual matrices to step through, matching the paper's
    description of the algorithm living in the weights.
    
    Args:
        seed: random seed for reproducibility
    Returns:
        dict of all weight matrices
    """
    rng = np.random.default_rng(seed)
    
    # --- Embedding matrix W_E: shape (D_MODEL, P) ---
    # Encodes sin(wk*a) and cos(wk*a) for each key frequency
    # We allocate 2 dimensions per key frequency (sin and cos)
    W_E = np.zeros((D_MODEL, P))
    for i, k in enumerate(KEY_FREQS):
        wk = 2.0 * np.pi * k / P
        for a in range(P):
            W_E[2*i, a] = np.cos(wk * a)      # cos component
            W_E[2*i + 1, a] = np.sin(wk * a)  # sin component
    # Add small noise to remaining dimensions (these get cleaned up by weight decay)
    W_E[2*N_KEY:, :] = rng.normal(0, 0.01, (D_MODEL - 2*N_KEY, P))
    
    # --- Positional embeddings: shape (3, D_MODEL) ---
    W_pos = rng.normal(0, 0.1, (3, D_MODEL))
    
    # --- Attention heads ---
    heads = []
    for j in range(N_HEADS):
        heads.append({
            'W_Q': rng.normal(0, 0.1, (D_HEAD, D_MODEL)),
            'W_K': rng.normal(0, 0.1, (D_HEAD, D_MODEL)),
            'W_V': rng.normal(0, 0.1, (D_HEAD, D_MODEL)),
            'W_O': rng.normal(0, 0.1, (D_MODEL, D_HEAD)),
        })
    
    # --- MLP weights ---
    # W_in: shape (D_MLP, D_MODEL), b_in: shape (D_MLP,)
    W_in = rng.normal(0, 0.1, (D_MLP, D_MODEL))
    b_in = np.zeros(D_MLP)
    # W_out: shape (D_MODEL, D_MLP), b_out: shape (D_MODEL,)
    W_out = rng.normal(0, 0.1, (D_MODEL, D_MLP))
    b_out = np.zeros(D_MODEL)
    
    # --- Unembedding matrix W_U: shape (P, D_MODEL) ---
    # Encodes cos(wk*c) and sin(wk*c) to read off the trig identities
    W_U = np.zeros((P, D_MODEL))
    for i, k in enumerate(KEY_FREQS):
        wk = 2.0 * np.pi * k / P
        for c in range(P):
            W_U[c, 2*i] = np.cos(wk * c)      # reads cos(wk*(a+b)) direction
            W_U[c, 2*i + 1] = np.sin(wk * c)  # reads sin(wk*(a+b)) direction
    
    return {
        'W_E': W_E,
        'W_pos': W_pos,
        'heads': heads,
        'W_in': W_in,
        'b_in': b_in,
        'W_out': W_out,
        'b_out': b_out,
        'W_U': W_U,
    }

def forward_transformer(a, b, weights):
    """
    Full forward pass through the simulated transformer.
    
    This follows the exact computation graph:
        1. Embed tokens a, b, '='
        2. Add positional embeddings
        3. Multi-head attention (from '=' to a, b)
        4. MLP with ReLU
        5. Unembed to logits
    
    Args:
        a: first input token, integer in [0, P-1]
        b: second input token, integer in [0, P-1]
        weights: dict from initialize_weights()
    Returns:
        predicted_c: argmax of logits
        logits: shape (P,)
        intermediates: dict of all intermediate activations
    """
    intermediates = {}
    
    # --- Step 1: Token embeddings ---
    emb_a = embed(a, weights['W_E'])    # shape (D_MODEL,)
    emb_b = embed(b, weights['W_E'])    # shape (D_MODEL,)
    emb_eq = np.zeros(D_MODEL)          # '=' token (simplified)
    intermediates['emb_a'] = emb_a
    intermediates['emb_b'] = emb_b
    
    # --- Step 2: Add positional embeddings ---
    x0_a = add_positional(emb_a, 0, weights['W_pos'])   # position 0
    x0_b = add_positional(emb_b, 1, weights['W_pos'])   # position 1
    x0_eq = add_positional(emb_eq, 2, weights['W_pos']) # position 2 ('=')
    intermediates['x0_a'] = x0_a
    intermediates['x0_b'] = x0_b
    intermediates['x0_eq'] = x0_eq
    
    # --- Step 3: Multi-head attention ---
    x1 = multi_head_attention(x0_a, x0_b, x0_eq, weights['heads'])
    intermediates['x1_post_attn'] = x1
    
    # --- Step 4: MLP ---
    mlp_output, hidden = mlp(x1, weights['W_in'], weights['b_in'],
                                  weights['W_out'], weights['b_out'])
    intermediates['mlp_hidden'] = hidden   # shape (D_MLP,) — the neuron activations
    intermediates['mlp_output'] = mlp_output
    
    # --- Step 4b: Residual connection around MLP ---
    # (Paper notes this is negligible, but we include it for completeness)
    x2 = x1 + mlp_output
    intermediates['x2_final'] = x2
    
    # --- Step 5: Unembed ---
    logits = unembed(x2, weights['W_U'])
    intermediates['logits'] = logits
    
    # --- Prediction ---
    predicted_c = np.argmax(logits)
    intermediates['predicted_c'] = predicted_c
    intermediates['true_c'] = (a + b) % P
    
    return predicted_c, logits, intermediates

# =============================================================================
# CONSTRUCTIVE INTERFERENCE DEMO
# =============================================================================
def demo_constructive_interference(a, b):
    """
    Visualize how summing cos(wk*(a+b-c)) over key frequencies
    creates a sharp peak at c* = (a+b) mod P.
    
    This is the core insight: individual cosines have many near-peaks,
    but their sum has a single dominant peak due to constructive interference.
    
    Args:
        a, b: input tokens
    Returns:
        logits: shape (P,) showing the interference pattern
    """
    true_c = (a + b) % P
    logits = np.zeros(P)
    
    print(f"\n{'='*60}")
    print(f"CONSTRUCTIVE INTERFERENCE DEMO: a={a}, b={b}, true c={true_c}")
    print(f"{'='*60}")
    
    for c in range(P):
        for k_idx, k in enumerate(KEY_FREQS):
            wk = 2.0 * np.pi * k / P
            contribution = np.cos(wk * (a + b - c))
            logits[c] += contribution
    
    # Show that the peak is at the correct answer
    peak_c = np.argmax(logits)
    print(f"  Peak logit at c={peak_c}, value={logits[peak_c]:.4f}")
    print(f"  True answer c*={true_c}, logit={logits[true_c]:.4f}")
    print(f"  Max non-answer logit: {np.max(np.delete(logits, true_c)):.4f}")
    print(f"  Ratio (peak / 2nd best): {logits[true_c] / np.sort(logits)[-2]:.2f}x")
    
    return logits

# =============================================================================
# MAIN: Run everything and print results
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GROKKING TRANSFORMER SIMULATION")
    print("Fourier Multiplication Algorithm for Modular Addition (mod 113)")
    print("=" * 70)
    
    # --- Test inputs ---
    a, b = 37, 89
    true_c = (a + b) % P
    print(f"\nInput: a={a}, b={b}")
    print(f"True answer: ({a} + {b}) mod {P} = {true_c}")
    
    # =========================================================================
    # TEST 1: Idealized forward pass (pure Fourier algorithm)
    # =========================================================================
    print(f"\n{'─'*70}")
    print("TEST 1: IDEALIZED FOURIER MULTIPLICATION (analytical)")
    print(f"{'─'*70}")
    
    pred, logits, info = forward_idealized(a, b)
    
    print(f"\n  Step 1 — Fourier Embedding of a={a}:")
    print(f"    cos(wk*a) = {info['cos_a']}")
    print(f"    sin(wk*a) = {info['sin_a']}")
    
    print(f"\n  Step 1 — Fourier Embedding of b={b}:")
    print(f"    cos(wk*b) = {info['cos_b']}")
    print(f"    sin(wk*b) = {info['sin_b']}")
    
    print(f"\n  Step 2 — Trig Identity (Attention + MLP compute a+b):")
    print(f"    cos(wk*(a+b)) = {info['cos_ab']}")
    print(f"    sin(wk*(a+b)) = {info['sin_ab']}")
    
    # Verify: these should equal cos/sin of wk*(a+b) directly
    direct_cos = np.cos(W_KEY * (a + b))
    direct_sin = np.sin(W_KEY * (a + b))
    print(f"    [Verification] Direct cos(wk*{a+b}) = {direct_cos}")
    print(f"    [Verification] Direct sin(wk*{a+b}) = {direct_sin}")
    print(f"    Match: {np.allclose(info['cos_ab'], direct_cos)}")
    
    print(f"\n  Step 3 — Logits (top 5):")
    top5 = np.argsort(logits)[-5:][::-1]
    for c in top5:
        print(f"    c={c:3d}: logit={logits[c]:8.3f}  {'← CORRECT' if c == true_c else ''}")
    
    
    print(f"\n  Step 4 — Prediction:")
    print(f"    Predicted c = {pred}")
    print(f"    True c      = {true_c}")
    print(f"    Correct: {pred == true_c}")
    print(f"    P(true c)   = {info['prob_true']:.6f}")
    
    # =========================================================================
    # TEST 2: Constructive interference demonstration
    # =========================================================================
    demo_constructive_interference(a, b)
    
    # =========================================================================
    # TEST 3: Full transformer simulation (with weight matrices)
    # =========================================================================
    print(f"\n{'─'*70}")
    print("TEST 3: SIMULATED TRANSFORMER (with actual weight matrices)")
    print(f"{'─'*70}")
    
    weights = initialize_weights(seed=42)
    pred_t, logits_t, info_t = forward_transformer(a, b, weights)
    
    print(f"\n  Embedding shapes:")
    print(f"    W_E:   {weights['W_E'].shape}  (D_MODEL x P)")
    print(f"    W_pos: {weights['W_pos'].shape}  (3 x D_MODEL)")
    print(f"    W_U:   {weights['W_U'].shape}  (P x D_MODEL)")
    
    print(f"\n  Residual stream after embedding + positional:")
    print(f"    x0_a norm: {np.linalg.norm(info_t['x0_a']):.4f}")
    print(f"    x0_b norm: {np.linalg.norm(info_t['x0_b']):.4f}")
    print(f"    x0_eq norm: {np.linalg.norm(info_t['x0_eq']):.4f}")
    
    print(f"\n  After multi-head attention:")
    print(f"    x1 norm: {np.linalg.norm(info_t['x1_post_attn']):.4f}")
    
    print(f"\n  MLP hidden layer (ReLU activations):")
    print(f"    Shape: {info_t['mlp_hidden'].shape}")
    print(f"    Active neurons (>0): {np.sum(info_t['mlp_hidden'] > 0)} / {D_MLP}")
    print(f"    Max activation: {np.max(info_t['mlp_hidden']):.4f}")
    print(f"    Mean (non-zero): {info_t['mlp_hidden'][info_t['mlp_hidden'] > 0].mean():.4f}")
    
    print(f"\n  Final residual stream:")
    print(f"    x2 norm: {np.linalg.norm(info_t['x2_final']):.4f}")
    
    print(f"\n  Logits (top 5):")
    top5_t = np.argsort(logits_t)[-5:][::-1]
    for c in top5_t:
        marker = '← CORRECT' if c == true_c else ''
        print(f"    c={c:3d}: logit={logits_t[c]:8.3f}  {marker}")
    
    print(f"\n  Note: The simulated transformer uses random (non-trained) weights")
    print(f"  for attention/MLP, so it won't get the right answer.")
    print(f"  The IDEALIZED version (Test 1) shows the correct algorithm.")
    
    # =========================================================================
    # TEST 4: Exhaustive correctness check of idealized algorithm
    # =========================================================================
    print(f"\n{'─'*70}")
    print("TEST 4: EXHAUSTIVE CORRECTNESS CHECK (all P*P = 12769 inputs)")
    print(f"{'─'*70}")
    
    correct_count = 0
    total = P * P
    
    for test_a in range(P):
        for test_b in range(P):
            pred_c, _, _ = forward_idealized(test_a, test_b)
            if pred_c == (test_a + test_b) % P:
                correct_count += 1
    
    accuracy = correct_count / total * 100
    print(f"\n  Results: {correct_count}/{total} correct ({accuracy:.2f}%)")
    print(f"  The idealized Fourier multiplication algorithm achieves perfect accuracy!")
    
    # =========================================================================
    # TEST 5: Demonstrate how individual neurons work
    # =========================================================================
    print(f"\n{'─'*70}")
    print("TEST 5: SINGLE NEURON ANALYSIS (how trig identities emerge from ReLU)")
    print(f"{'─'*70}")
    
    # Simulate a single "ideal" MLP neuron that computes cos(wk*(a+b))
    # The neuron's pre-activation is: cos(wk*a)*cos(wk*b) - sin(wk*a)*sin(wk*b) + bias
    # After ReLU, pairs of neurons with opposite phases reconstruct the full cosine.
    
    k_demo = KEY_FREQS[0]  # frequency 14
    wk_demo = 2.0 * np.pi * k_demo / P
    
    print(f"\n  Demonstrating neuron for frequency k={k_demo} (wk={wk_demo:.4f})")
    print(f"  Input: a={a}, b={b}")
    
    # What the neuron computes (pre-ReLU):
    cos_a = np.cos(wk_demo * a)
    cos_b = np.cos(wk_demo * b)
    sin_a = np.sin(wk_demo * a)
    sin_b = np.sin(wk_demo * b)
    
    # The attention layer produces products like cos(wk*a)*cos(wk*b)
    # via bilinear interaction (attention weight * OV circuit)
    term_cc = cos_a * cos_b
    term_ss = sin_a * sin_b
    term_sc = sin_a * cos_b
    term_cs = cos_a * sin_b
    
    print(f"\n  Products computed by attention (degree-2 polynomials):")
    print(f"    cos(wk*a)*cos(wk*b) = {cos_a:.4f} * {cos_b:.4f} = {term_cc:.4f}")
    print(f"    sin(wk*a)*sin(wk*b) = {sin_a:.4f} * {sin_b:.4f} = {term_ss:.4f}")
    print(f"    sin(wk*a)*cos(wk*b) = {sin_a:.4f} * {cos_b:.4f} = {term_sc:.4f}")
    print(f"    cos(wk*a)*sin(wk*b) = {cos_a:.4f} * {sin_b:.4f} = {term_cs:.4f}")
    
    # Trig identity: cos(wk*(a+b)) = cos*cos - sin*sin
    cos_ab_computed = term_cc - term_ss
    sin_ab_computed = term_sc + term_cs
    cos_ab_direct = np.cos(wk_demo * (a + b))
    sin_ab_direct = np.sin(wk_demo * (a + b))
    
    print(f"\n  Trig identity verification:")
    print(f"    cos(wk*(a+b)) via identity: {cos_ab_computed:.6f}")
    print(f"    cos(wk*(a+b)) direct:       {cos_ab_direct:.6f}")
    print(f"    Match: {np.isclose(cos_ab_computed, cos_ab_direct)}")
    print(f"    sin(wk*(a+b)) via identity: {sin_ab_computed:.6f}")
    print(f"    sin(wk*(a+b)) direct:       {sin_ab_direct:.6f}")
    print(f"    Match: {np.isclose(sin_ab_computed, sin_ab_direct)}")
    
    # Show how ReLU + pairs of neurons reconstruct the cosine:
    # Neuron+ has pre-activation: cos(wk*(a+b)) + bias (bias ~ 0.5 to shift into ReLU range)
    # Neuron- has pre-activation: -cos(wk*(a+b)) + bias
    # ReLU(x+0.5) - ReLU(-x+0.5) ≈ x for |x| < 0.5
    bias = 1.0  # Large enough that most values pass through ReLU
    neuron_pos = relu(cos_ab_computed + bias)
    neuron_neg = relu(-cos_ab_computed + bias)
    reconstructed = neuron_pos - neuron_neg
    
    print(f"\n  ReLU reconstruction of cos(wk*(a+b)):")
    print(f"    Neuron+ pre-act: {cos_ab_computed + bias:.4f} → ReLU: {neuron_pos:.4f}")
    print(f"    Neuron- pre-act: {-cos_ab_computed + bias:.4f} → ReLU: {neuron_neg:.4f}")
    print(f"    Reconstructed (pos - neg): {reconstructed:.4f}")
    print(f"    True value: {cos_ab_computed:.4f}")
    print(f"    (In practice, the model uses ~44 neurons per frequency for robustness)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY: THE FOURIER MULTIPLICATION ALGORITHM")
    print(f"{'='*70}")
    print(f"""
    Given inputs a={a}, b={b}, computing (a+b) mod {P} = {true_c}:
    
    1. EMBED: Map a,b → sin(wk*a), cos(wk*a), sin(wk*b), cos(wk*b)
       for k ∈ {KEY_FREQS}
       
    2. ATTENTION + MLP: Compute trig identities
       cos(wk*(a+b)) = cos(wk*a)cos(wk*b) - sin(wk*a)sin(wk*b)
       sin(wk*(a+b)) = sin(wk*a)cos(wk*b) + cos(wk*a)sin(wk*b)
       
    3. UNEMBED: For each candidate c, compute
       logit(c) = Σ_k α_k * cos(wk*(a+b-c))
       
    4. CONSTRUCTIVE INTERFERENCE:
       At c* = {true_c} = ({a}+{b}) mod {P}, ALL cosines = 1
       → logit({true_c}) = Σ α_k = {sum([44.1, 42.2, 44.8, 66.6, 63.0]):.1f} (maximum!)
       At other c values, cosines partially cancel → smaller logits
       
    Key insight: {N_KEY} frequencies suffice because their sum has a unique
    global maximum at 0 mod {P} (constructive interference), while individual
    cosines have many near-peaks that cancel when summed (destructive interference).
    """)
    
    print("Done! Set breakpoints anywhere above to inspect intermediate values.")
    print("Try: python -m pdb grokking_transformer_sim.py")
