import torch
import torch.nn as nn
import torch.nn.functional as F


class TabMixerBlock(nn.Module):
    def __init__(self, num_features: int, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + attn_out

        residual2 = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        x = residual2 + mlp_out
        return x

class TabMixer(nn.Module):
    def __init__(
        self,
        num_features: int,
        dim: int = 64,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()
        self.feature_embed = nn.Linear(1, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TabMixerBlock(num_features=num_features, dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(num_features * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = x.size()
        x = x.unsqueeze(-1)               # (B, F, 1)
        x = self.feature_embed(x)         # (B, F, dim)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)                  # (B, F, dim)
        x = x.reshape(batch_size, -1)     # (B, F*dim)
        out = self.head(x)                # (B, num_classes==dim)
        return out                        # → (B, hidden_node)


class CQLContextGatedFusionMixerNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_node: int,
        activation: str = 'relu',
        note_emb_dim: int = 768, # or 4096
    ):
        super().__init__()
        self.note_proj1 = nn.Linear(note_emb_dim, hidden_node)
        self.note_bn1   = nn.BatchNorm1d(hidden_node)
        self.note_proj2 = nn.Linear(hidden_node, hidden_node)
        self.note_bn2   = nn.BatchNorm1d(hidden_node)
        self.ctx_proj1  = nn.Linear(note_emb_dim, hidden_node)
        self.ctx_bn1    = nn.BatchNorm1d(hidden_node)
        self.ctx_proj2  = nn.Linear(hidden_node, hidden_node)
        self.ctx_bn2    = nn.BatchNorm1d(hidden_node)


        self.state_encoder = TabMixer(
            num_features=state_dim,
            dim=hidden_node,
            depth=4,         
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.1,
            num_classes=hidden_node
        )

        # Gated fusion
        self.gate = nn.Linear(hidden_node * 2, hidden_node)
        # Bidirectional cross-attention
        self.cross_attn_st2tx = nn.MultiheadAttention(embed_dim=hidden_node, num_heads=4)
        self.cross_attn_tx2st = nn.MultiheadAttention(embed_dim=hidden_node, num_heads=4)

        # Combine & dueling
        self.combined_dim = hidden_node * 4
        self.share_fc   = nn.Linear(self.combined_dim, hidden_node)
        self.share_bn   = nn.BatchNorm1d(hidden_node)
        self.value_fc1 = nn.Linear(hidden_node, hidden_node)
        self.value_bn  = nn.BatchNorm1d(hidden_node)
        self.value_fc2 = nn.Linear(hidden_node, 1)
        self.adv_fc1   = nn.Linear(hidden_node, hidden_node)
        self.adv_bn    = nn.BatchNorm1d(hidden_node)
        self.adv_fc2   = nn.Linear(hidden_node, num_actions)

        if activation.lower() == 'relu':
            self.act = F.relu
        elif activation.lower() == 'tanh':
            self.act = torch.tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, state: torch.Tensor, note_emb: torch.Tensor, context_emb: torch.Tensor):
        x_note = self.act(self.note_bn1(self.note_proj1(note_emb)))
        x_note = self.act(self.note_bn2(self.note_proj2(x_note)))
        x_ctx  = self.act(self.ctx_bn1(self.ctx_proj1(context_emb)))
        x_ctx  = self.act(self.ctx_bn2(self.ctx_proj2(x_ctx)))

        # State → TabMixer
        x_state = self.state_encoder(state)  # (B, hidden_node)

        # Gated fusion
        gate_in    = torch.cat([x_note, x_ctx], dim=1)
        gate_score = torch.sigmoid(self.gate(gate_in))
        fused_text = gate_score * x_note + (1 - gate_score) * x_ctx

        # Cross-modal attention
        attn_st, _ = self.cross_attn_st2tx(
            x_state.unsqueeze(0), fused_text.unsqueeze(0), fused_text.unsqueeze(0)
        )
        attn_st = attn_st.squeeze(0)
        attn_tx, _ = self.cross_attn_tx2st(
            fused_text.unsqueeze(0), x_state.unsqueeze(0), x_state.unsqueeze(0)
        )
        attn_tx = attn_tx.squeeze(0)

        # residual+attn → concat
        state_feat = torch.cat([x_state, attn_tx], dim=1)
        text_feat  = torch.cat([fused_text, attn_st], dim=1)
        combined   = torch.cat([state_feat, text_feat], dim=1)

        # downstream
        x = self.act(self.share_bn(self.share_fc(combined)))
        v = self.value_fc2(self.act(self.value_bn(self.value_fc1(x))))
        a = self.adv_fc2(self.act(self.adv_bn(self.adv_fc1(x))))
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q


class BertNetCQL(nn.Module):
    def __init__(
        self,
        num_actions: int,
        hidden_node: int,
        activation: str = 'relu',
        note_emb_dim: int = 768
    ):
        super(BertNetCQL, self).__init__()
        # Note embedding projection
        self.note_proj1 = nn.Linear(note_emb_dim, hidden_node)
        self.note_bn1   = nn.BatchNorm1d(hidden_node)
        self.note_proj2 = nn.Linear(hidden_node, hidden_node)
        self.note_bn2   = nn.BatchNorm1d(hidden_node)

        # Shared feature branch
        self.share_q    = nn.Linear(hidden_node, hidden_node)
        self.share_bn   = nn.BatchNorm1d(hidden_node)

        # Dueling: Value branch
        self.value_q1   = nn.Linear(hidden_node, hidden_node)
        self.value_bn1  = nn.BatchNorm1d(hidden_node)
        self.value_q2   = nn.Linear(hidden_node, 1)
        self.value_bn2  = nn.BatchNorm1d(1)

        # Dueling: Advantage branch
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
        # embedding projection
        x = self.act(self.note_bn1(self.note_proj1(note_emb)))
        x = self.act(self.note_bn2(self.note_proj2(x)))

        # shared features
        x = self.act(self.share_bn(self.share_q(x)))

        # value stream
        v = self.act(self.value_bn1(self.value_q1(x)))
        v = self.act(self.value_bn2(self.value_q2(v)))  # [B,1]

        # advantage stream
        adv = self.act(self.adv_bn1(self.adv_q1(x)))
        adv = self.act(self.adv_bn2(self.adv_q2(adv))) # [B,A]

        # dueling combine
        adv_mean = adv.mean(dim=1, keepdim=True)       # [B,1]
        q = v + adv - adv_mean                         # [B,A]
        return q



class CQLNet(torch.nn.Module):
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