import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class FeatureMixerBlock(nn.Module):
    """
    MLP-Mixer style feature-mixing block for hidden-dimension state data.
    Only mixes across feature dimension to avoid batch-size dependence.
    Outputs maintain hidden_dim size.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # feature-mixing across feature dimension
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch, hidden_dim]
        residual = x
        v = self.feature_mlp(x)
        x = residual + v
        x = self.norm(x)
        return x
    

class CQLContextGatedFusionMixerNet(nn.Module):

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_node: int,
        activation: str = "relu",
        note_emb_dim: int = 768,
        num_heads: int = 4,
        init_q_scale: float = 0.02,
        attn_entropy_coef: float = 0.0,   # NEW: entropy penalty coefficient (>=0)
        entropy_eps: float = 1e-8,        # NEW: log 안정화
    ):
        super().__init__()

        # -----------------------------
        # Note projection
        # -----------------------------
        self.note_proj1 = nn.Linear(note_emb_dim, hidden_node)
        self.note_bn1   = nn.BatchNorm1d(hidden_node)
        self.note_proj2 = nn.Linear(hidden_node, hidden_node)
        self.note_bn2   = nn.BatchNorm1d(hidden_node)

        # -----------------------------
        # Context projection
        # -----------------------------
        self.ctx_proj1  = nn.Linear(note_emb_dim, hidden_node)
        self.ctx_bn1    = nn.BatchNorm1d(hidden_node)
        self.ctx_proj2  = nn.Linear(hidden_node, hidden_node)
        self.ctx_bn2    = nn.BatchNorm1d(hidden_node)

        # -----------------------------
        # State projection + mixers
        # -----------------------------
        self.state_proj = nn.Linear(state_dim, hidden_node)
        self.state_bn0  = nn.BatchNorm1d(hidden_node)
        self.state_mixer1 = FeatureMixerBlock(hidden_dim=hidden_node)
        self.state_mixer2 = FeatureMixerBlock(hidden_dim=hidden_node)

        # -----------------------------
        # Gated fusion (note vs context)
        # -----------------------------
        self.gate = nn.Linear(hidden_node * 2, hidden_node)

        # -----------------------------
        # 2-token attention
        # -----------------------------
        self.attn_state_query = nn.MultiheadAttention(embed_dim=hidden_node, num_heads=num_heads)
        self.attn_text_query  = nn.MultiheadAttention(embed_dim=hidden_node, num_heads=num_heads)

        # learnable router queries
        self.q_state_tok = nn.Parameter(torch.randn(1, 1, hidden_node) * init_q_scale)
        self.q_text_tok  = nn.Parameter(torch.randn(1, 1, hidden_node) * init_q_scale)

        # -----------------------------
        # Combine -> shared -> dueling
        # -----------------------------
        self.combined_dim = hidden_node * 4
        self.share_fc = nn.Linear(self.combined_dim, hidden_node)
        self.share_bn = nn.BatchNorm1d(hidden_node)

        self.value_fc1 = nn.Linear(hidden_node, hidden_node)
        self.value_bn  = nn.BatchNorm1d(hidden_node)
        self.value_fc2 = nn.Linear(hidden_node, 1)

        self.adv_fc1 = nn.Linear(hidden_node, hidden_node)
        self.adv_bn  = nn.BatchNorm1d(hidden_node)
        self.adv_fc2 = nn.Linear(hidden_node, num_actions)

        # -----------------------------
        # Activation
        # -----------------------------
        if activation.lower() == "relu":
            self.act = F.relu
        elif activation.lower() == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # -----------------------------
        # NEW: entropy penalty configs
        # -----------------------------
        self.attn_entropy_coef = float(attn_entropy_coef)
        self.entropy_eps = float(entropy_eps)

    def _attn_entropy_from_weights(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: (B, heads, 1, 2)  (average_attn_weights=False)
        return: scalar entropy (mean over B and heads)
        """
        # (B, heads, 2)
        p = w.squeeze(2)
        # entropy over last dim (2-token distribution)
        ent = -(p * (p + self.entropy_eps).log()).sum(dim=-1)  # (B, heads)
        return ent.mean()  # scalar

    def forward(
        self,
        state: torch.Tensor,
        note_emb: torch.Tensor,
        context_emb: torch.Tensor,
        return_attn: bool = False,
    ):
        # -----------------------------
        # Project note & context
        # -----------------------------
        x_note = self.act(self.note_bn1(self.note_proj1(note_emb)))
        x_note = self.act(self.note_bn2(self.note_proj2(x_note)))

        x_ctx  = self.act(self.ctx_bn1(self.ctx_proj1(context_emb)))
        x_ctx  = self.act(self.ctx_bn2(self.ctx_proj2(x_ctx)))

        # -----------------------------
        # Project state and apply feature mixers
        # -----------------------------
        x_state = self.act(self.state_bn0(self.state_proj(state)))
        x_state = self.state_mixer1(x_state)
        x_state = self.state_mixer2(x_state)

        # -----------------------------
        # Gated fusion
        # -----------------------------
        gate_in    = torch.cat([x_note, x_ctx], dim=1)          # (B, 2H)
        gate       = torch.sigmoid(self.gate(gate_in))          # (B, H)
        fused_text = gate * x_note + (1.0 - gate) * x_ctx       # (B, H)

        # KV = [state, text]
        kv = torch.stack([x_state, fused_text], dim=0)          # (2, B, H)
        B  = x_state.size(0)

        # Q = learnable tokens
        q_state = self.q_state_tok.expand(1, B, -1)             # (1, B, H)
        q_text  = self.q_text_tok.expand(1, B, -1)              # (1, B, H)

        # (1) state-router query
        out_s, w_s = self.attn_state_query(
            q_state, kv, kv,
            need_weights=True,
            average_attn_weights=False,
        )
        out_s = out_s.squeeze(0)                                # (B, H)

        # (2) text-router query
        out_t, w_t = self.attn_text_query(
            q_text, kv, kv,
            need_weights=True,
            average_attn_weights=False,
        )
        out_t = out_t.squeeze(0)                                # (B, H)

        # head mean -> (B,2)
        w_s_mean = w_s.mean(dim=1).squeeze(1)                   # (B,2)
        w_t_mean = w_t.mean(dim=1).squeeze(1)                   # (B,2)

        attn_s_to_state = w_s_mean[:, 0]
        attn_s_to_text  = w_s_mean[:, 1]
        attn_t_to_state = w_t_mean[:, 0]
        attn_t_to_text  = w_t_mean[:, 1]

        # -----------------------------
        # NEW: entropy penalty (low entropy => more decisive)
        # -----------------------------
        # raw entropies (scalar)
        ent_s = self._attn_entropy_from_weights(w_s)  # scalar
        ent_t = self._attn_entropy_from_weights(w_t)  # scalar
        attn_entropy = 0.5 * (ent_s + ent_t)          # scalar
        attn_entropy_penalty = self.attn_entropy_coef * attn_entropy  # scalar

        # -----------------------------
        # Build embedding
        # -----------------------------
        emb_state = torch.cat([x_state, out_s], dim=1)          # (B, 2H)
        emb_text  = torch.cat([fused_text, out_t], dim=1)       # (B, 2H)
        combined  = torch.cat([emb_state, emb_text], dim=1)     # (B, 4H)

        x = self.act(self.share_bn(self.share_fc(combined)))

        # -----------------------------
        # Dueling
        # -----------------------------
        v = self.act(self.value_bn(self.value_fc1(x)))
        v = self.value_fc2(v)

        a = self.act(self.adv_bn(self.adv_fc1(x)))
        a = self.adv_fc2(a)

        q = v + (a - a.mean(dim=1, keepdim=True))

        if return_attn:
            aux = {
                # gate
                "gate_mean": gate.mean(dim=1).detach(),
                "gate_vec":  gate.detach(),

                # router attention (mean over heads)
                "attn_s_to_state": attn_s_to_state.detach(),
                "attn_s_to_text":  attn_s_to_text.detach(),
                "attn_t_to_state": attn_t_to_state.detach(),
                "attn_t_to_text":  attn_t_to_text.detach(),

                # raw head weights
                "w_s": w_s.detach(),
                "w_t": w_t.detach(),

                # learnable queries
                "q_state_tok": self.q_state_tok.detach().clone(),
                "q_text_tok":  self.q_text_tok.detach().clone(),

                # NEW: entropy stats (로깅용 detach)
                "attn_entropy": attn_entropy.detach(),
                "attn_entropy_s": ent_s.detach(),
                "attn_entropy_t": ent_t.detach(),

                "attn_entropy_penalty": attn_entropy_penalty,
            }
            return q, aux

        return q
    

class TextNetCQL(nn.Module):
    def __init__(
        self,
        num_actions: int,
        hidden_node: int,
        activation: str = 'relu',
        note_emb_dim: int = 768
    ):
        super(TextNetCQL, self).__init__()
        # --- Note embedding projection ---
        self.note_proj1 = nn.Linear(note_emb_dim, hidden_node)
        self.note_bn1   = nn.BatchNorm1d(hidden_node)
        self.note_proj2 = nn.Linear(hidden_node, hidden_node)
        self.note_bn2   = nn.BatchNorm1d(hidden_node)

        # --- Shared feature branch ---
        self.share_q    = nn.Linear(hidden_node, hidden_node)
        self.share_bn   = nn.BatchNorm1d(hidden_node)

        # --- Dueling: Value branch ---
        self.value_q1   = nn.Linear(hidden_node, hidden_node)
        self.value_bn1  = nn.BatchNorm1d(hidden_node)
        self.value_q2   = nn.Linear(hidden_node, 1)
        self.value_bn2  = nn.BatchNorm1d(1)

        # --- Dueling: Advantage branch ---
        self.adv_q1     = nn.Linear(hidden_node, hidden_node)
        self.adv_bn1    = nn.BatchNorm1d(hidden_node)
        self.adv_q2     = nn.Linear(hidden_node, num_actions)
        self.adv_bn2    = nn.BatchNorm1d(num_actions)

        # Activation
        if activation.lower() == 'relu':
            self.act = F.relu
        elif activation.lower() == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, note_emb: torch.Tensor) -> torch.Tensor:
        # 1) embedding projection
        x = self.act(self.note_bn1(self.note_proj1(note_emb)))
        x = self.act(self.note_bn2(self.note_proj2(x)))

        # 2) shared features
        x = self.act(self.share_bn(self.share_q(x)))

        # 3) value stream
        v = self.act(self.value_bn1(self.value_q1(x)))
        v = self.act(self.value_bn2(self.value_q2(v)))  # [B,1]

        # 4) advantage stream
        adv = self.act(self.adv_bn1(self.adv_q1(x)))
        adv = self.act(self.adv_bn2(self.adv_q2(adv))) # [B,A]

        # 5) dueling combine
        adv_mean = adv.mean(dim=1, keepdim=True)       # [B,1]
        q = v + adv - adv_mean                         # [B,A]
        return q


# class IQLActor(nn.Module):
#     """Actor (Policy) Model."""

#     def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
#         super(IQLActor, self).__init__()
       
#         self.fc1 = nn.Linear(state_dim, hidden_node)
#         self.bn1 = nn.BatchNorm1d(num_features=hidden_node)
#         self.fc2 = nn.Linear(hidden_node, hidden_node)
#         self.bn2 = nn.BatchNorm1d(num_features=hidden_node)
#         self.fc3 = nn.Linear(hidden_node, num_actions)

#     def forward(self, state):

#         x = F.relu(self.bn1(self.fc1(state)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         action_logits = self.fc3(x)
#         return action_logits
    
    


# class IQLCritic(nn.Module):
#     """Dueling IQL Critic using BCQNet-style layers."""
#     def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
#         super(IQLCritic, self).__init__()
#         # shared backbone
#         self.share_q = nn.Linear(state_dim, hidden_node)
#         self.share_bn = nn.BatchNorm1d(num_features=hidden_node)
        
#         # value stream
#         self.value_q1 = nn.Linear(hidden_node, hidden_node)
#         self.value_bn1 = nn.BatchNorm1d(num_features=hidden_node)
#         self.value_q2 = nn.Linear(hidden_node, 1)
#         self.value_bn2 = nn.BatchNorm1d(num_features=1)
        
#         # advantage stream
#         self.adv_q1 = nn.Linear(hidden_node, hidden_node)
#         self.adv_bn1 = nn.BatchNorm1d(num_features=hidden_node)
#         self.adv_q2 = nn.Linear(hidden_node, num_actions)
#         self.adv_bn2 = nn.BatchNorm1d(num_features=num_actions)
        
#         # activation 
#         if activation.lower() == 'relu':
#             self.activation = F.relu
#         elif activation.lower() == 'tanh':
#             self.activation = F.tanh
#         else:
#             raise ValueError(f"Unsupported activation: {activation}")

#     def forward(self, state):
#         # shared feature encoding
#         x = self.activation(self.share_bn(self.share_q(state)))
        
#         # value head
#         v = self.activation(self.value_bn1(self.value_q1(x)))
#         v = self.activation(self.value_bn2(self.value_q2(v)))  # shape [B,1]
        
#         # advantage head
#         a = self.activation(self.adv_bn1(self.adv_q1(x)))
#         a = self.activation(self.adv_bn2(self.adv_q2(a)))      # shape [B,A]
        
#         # dueling combine
#         a_mean = a.mean(dim=1, keepdim=True)                  # shape [B,1]
#         q = v + (a - a_mean)                                   # shape [B,A]
#         return q
    

# class IQLValue(nn.Module):
#     """Value network in BCQNet style (no dueling)."""
#     def __init__(self, state_dim, hidden_node, activation='relu'):
#         super(IQLValue, self).__init__()
#         # shared encoder (same as BCQNet share layers)
#         self.share_q = nn.Linear(state_dim, hidden_node)
#         self.share_bn = nn.BatchNorm1d(num_features=hidden_node)
        
#         # value stream (reuse BCQNet’s value layers)
#         self.value_q1 = nn.Linear(hidden_node, hidden_node)
#         self.value_bn1 = nn.BatchNorm1d(num_features=hidden_node)
#         self.value_q2 = nn.Linear(hidden_node, 1)
#         self.value_bn2 = nn.BatchNorm1d(num_features=1)
        
#         # activation 
#         if activation.lower() == 'relu':
#             self.activation = F.relu
#         elif activation.lower() == 'tanh':
#             self.activation = F.tanh
#         else:
#             raise ValueError(f"Unsupported activation: {activation}")

#     def forward(self, state):
#         # shared encoding
#         x = self.activation(self.share_bn(self.share_q(state)))
#         # value head
#         x = self.activation(self.value_bn1(self.value_q1(x)))
#         v = self.value_bn2(self.value_q2(x))  
#                                              
#         return v

class CQLNet(torch.nn.Module):
    """Dueling Q-network with Batch Normalization for discrete actions"""
    def __init__(self, state_dim, num_actions, hidden_node, activation='relu'):
        super(CQLNet, self).__init__()
        # Shared layers
        self.fc1 = torch.nn.Linear(state_dim, hidden_node)
        self.bn1 = torch.nn.BatchNorm1d(hidden_node)
        self.fc2 = torch.nn.Linear(hidden_node, hidden_node)
        self.bn2 = torch.nn.BatchNorm1d(hidden_node)

        # Value stream
        self.value_fc = torch.nn.Linear(hidden_node, hidden_node)
        self.value_bn = torch.nn.BatchNorm1d(hidden_node)
        self.value_out = torch.nn.Linear(hidden_node, 1)

        # Advantage stream
        self.adv_fc = torch.nn.Linear(hidden_node, hidden_node)
        self.adv_bn = torch.nn.BatchNorm1d(hidden_node)
        self.adv_out = torch.nn.Linear(hidden_node, num_actions)

        # Activation
        if activation.lower() == 'relu':
            self.act = F.relu
        elif activation.lower() == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError("Unsupported activation")

    def forward(self, state):
        # Shared
        x = self.fc1(state)
        x = self.bn1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)

        # Value
        v = self.value_fc(x)
        v = self.value_bn(v)
        v = self.act(v)
        v = self.value_out(v)

        # Advantage
        a = self.adv_fc(x)
        a = self.adv_bn(a)
        a = self.act(a)
        a = self.adv_out(a)

        # Combine streams
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q