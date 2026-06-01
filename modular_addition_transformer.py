import math

import torch
import torch.nn as nn

class ModularAdditionTransformer(nn.Module):
    """
    1-layer transformer for modular addition, following Nanda et al. (2023).
    Input: "a b =" -> predicts (a+b) mod P
    """
    def __init__(self, P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp

        # Embeddings
        self.embed = nn.Embedding(P + 1, d_model)  # P tokens + 1 for '='
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions: a, b, =

        # Attention (single layer)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # MLP
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)

        # Unembed
        self.unembed = nn.Linear(d_model, P, bias=False)

    def forward(self, a_idx, b_idx):
        """Forward pass. a_idx, b_idx are integer tensors."""
        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)

        # Embed tokens + positions
        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)
        x = self.embed(tok_ids) + self.pos_embed(pos_ids)  # (batch, 3, d_model)

        # Attention (from position 2 '=' to positions 0,1)
        Q = self.W_Q(x[:, 2:3, :])  # (batch, 1, d_model)
        K = self.W_K(x[:, :2, :])   # (batch, 2, d_model)
        V = self.W_V(x[:, :2, :])   # (batch, 2, d_model)

        # Multi-head attention
        batch_size = Q.shape[0]
        Q = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)  # (batch, n_heads, 1, 2)
        attn_out = torch.matmul(attn, V)  # (batch, n_heads, 1, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)

        # Residual + MLP
        residual = x[:, 2:3, :] + attn_out  # (batch, 1, d_model)
        mlp_hidden = F.relu(self.mlp_in(residual))
        mlp_out = self.mlp_out(mlp_hidden)
        final = residual + mlp_out  # (batch, 1, d_model)

        logits = self.unembed(final.squeeze(1))  # (batch, P)
        return logits

    def forward_with_hooks(self, a_idx, b_idx, hook_points=None):
        """
        Forward pass that returns intermediate activations at specified hook points.
        Used for ACDC-style activation patching.

        hook_points: dict mapping hook_name -> None (will be filled with activations)
        Returns: logits, activations_dict
        """
        if hook_points is None:
            hook_points = {}

        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)

        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)

        # Embedding
        tok_embed = self.embed(tok_ids)
        pos_embed_val = self.pos_embed(pos_ids)
        x = tok_embed + pos_embed_val

        activations = {}
        activations["embed"] = x.detach().clone()
        activations["tok_embed"] = tok_embed.detach().clone()
        activations["pos_embed"] = pos_embed_val.detach().clone()

        # Attention
        Q = self.W_Q(x[:, 2:3, :])
        K = self.W_K(x[:, :2, :])
        V = self.W_V(x[:, :2, :])

        activations["Q"] = Q.detach().clone()
        activations["K"] = K.detach().clone()
        activations["V"] = V.detach().clone()

        batch_size = Q.shape[0]
        Q_heads = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K_heads = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V_heads = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)
        activations["attn_weights"] = attn_weights.detach().clone()

        attn_out = torch.matmul(attn_weights, V_heads)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)
        activations["attn_out"] = attn_out.detach().clone()

        # Per-head attention outputs
        for h in range(self.n_heads):
            V_h = V_heads[:, h:h+1, :, :]  # (batch, 1, 2, d_head)
            attn_h = attn_weights[:, h:h+1, :, :]  # (batch, 1, 1, 2)
            out_h = torch.matmul(attn_h, V_h)  # (batch, 1, 1, d_head)
            activations[f"attn_head_{h}"] = out_h.squeeze(1).detach().clone()

        # Residual stream after attention
        residual = x[:, 2:3, :] + attn_out
        activations["residual_mid"] = residual.detach().clone()

        # MLP
        mlp_pre = self.mlp_in(residual)
        activations["mlp_pre"] = mlp_pre.detach().clone()
        mlp_hidden = F.relu(mlp_pre)
        activations["mlp_hidden"] = mlp_hidden.detach().clone()
        mlp_out = self.mlp_out(mlp_hidden)
        activations["mlp_out"] = mlp_out.detach().clone()

        # Final
        final = residual + mlp_out
        activations["residual_final"] = final.detach().clone()

        logits = self.unembed(final.squeeze(1))
        activations["logits"] = logits.detach().clone()

        return logits, activations

    def forward_with_patches(self, a_idx, b_idx, patches: dict):
        """
        Forward pass with activation patching applied.
        patches: dict mapping hook_name -> replacement_tensor
        Replaces the activation at the specified hook point with the given tensor.
        """
        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)

        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)

        tok_embed = self.embed(tok_ids)
        pos_embed_val = self.pos_embed(pos_ids)

        if "tok_embed" in patches:
            tok_embed = patches["tok_embed"]
        if "pos_embed" in patches:
            pos_embed_val = patches["pos_embed"]

        x = tok_embed + pos_embed_val
        if "embed" in patches:
            x = patches["embed"]

        # Attention
        Q = self.W_Q(x[:, 2:3, :])
        K = self.W_K(x[:, :2, :])
        V = self.W_V(x[:, :2, :])

        if "Q" in patches:
            Q = patches["Q"]
        if "K" in patches:
            K = patches["K"]
        if "V" in patches:
            V = patches["V"]

        batch_size = Q.shape[0]
        Q_heads = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K_heads = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V_heads = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        # Per-head patching
        for h in range(self.n_heads):
            if f"Q_head_{h}" in patches:
                Q_heads[:, h, :, :] = patches[f"Q_head_{h}"]
            if f"K_head_{h}" in patches:
                K_heads[:, h, :, :] = patches[f"K_head_{h}"]
            if f"V_head_{h}" in patches:
                V_heads[:, h, :, :] = patches[f"V_head_{h}"]

        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_weights = torch.softmax(scores, dim=-1)

        if "attn_weights" in patches:
            attn_weights = patches["attn_weights"]

        attn_out = torch.matmul(attn_weights, V_heads)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)

        if "attn_out" in patches:
            attn_out = patches["attn_out"]

        residual = x[:, 2:3, :] + attn_out
        if "residual_mid" in patches:
            residual = patches["residual_mid"]

        mlp_pre = self.mlp_in(residual)
        if "mlp_pre" in patches:
            mlp_pre = patches["mlp_pre"]

        mlp_hidden = F.relu(mlp_pre)
        if "mlp_hidden" in patches:
            mlp_hidden = patches["mlp_hidden"]

        mlp_out = self.mlp_out(mlp_hidden)
        if "mlp_out" in patches:
            mlp_out = patches["mlp_out"]

        final = residual + mlp_out
        if "residual_final" in patches:
            final = patches["residual_final"]

        logits = self.unembed(final.squeeze(1))
        return logits

    def get_mlp_activations(self, a_idx, b_idx):
        """Get MLP hidden activations for analysis."""
        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)
        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)
        x = self.embed(tok_ids) + self.pos_embed(pos_ids)

        Q = self.W_Q(x[:, 2:3, :])
        K = self.W_K(x[:, :2, :])
        V = self.W_V(x[:, :2, :])

        batch_size = Q.shape[0]
        Q = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)

        residual = x[:, 2:3, :] + attn_out
        mlp_hidden = F.relu(self.mlp_in(residual))
        return mlp_hidden.squeeze(1), attn  # (batch, d_mlp), (batch, n_heads, 1, 2)


