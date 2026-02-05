from buffer import ReplayBuffer
import os
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# Global config
# ============================================================
PTH_DIR = "./pth"


# ============================================================
# Utils
# ============================================================
def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def _to_2d_float_array(x) -> np.ndarray:
    x = _to_numpy(x)

    if isinstance(x, np.ndarray):
        if x.dtype == object:
            x = np.stack([np.asarray(_to_numpy(v), dtype=np.float32) for v in x], axis=0)
        else:
            x = x.astype(np.float32, copy=False)
    else:
        x = np.stack([np.asarray(_to_numpy(v), dtype=np.float32) for v in x], axis=0)

    if x.ndim == 3:
        x = x.mean(axis=1)

    if x.ndim != 2:
        raise ValueError(f"embedding must be 2D (N,D) (or 3D (N,T,D)), got shape={x.shape}")

    return x


def _get_note_embeddings_from_buffer(buf) -> np.ndarray:

    if not hasattr(buf, "note"):
        raise AttributeError("buf.note anticipate")

    note = buf.note[:buf.crt_size]
    return _to_2d_float_array(note)


def _load_first_existing(paths: list[str]):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ============================================================
# Models
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class NoteOnlyPolicy(nn.Module):
    def __init__(
        self,
        note_dim: int,
        n_actions: int,
        note_hidden: int = 256,
        note_latent: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.note_dim = note_dim
        self.n_actions = n_actions

        self.note_encoder = MLPEncoder(note_dim, note_latent, hidden_dim=note_hidden, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(note_latent, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, note_x):
        z = self.note_encoder(note_x)
        return self.head(z)


class LateFusionPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        note_dim: int,
        n_actions: int,
        state_hidden: int = 128,
        note_hidden: int = 256,
        state_latent: int = 64,
        note_latent: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.note_dim = note_dim
        self.n_actions = n_actions

        self.state_encoder = MLPEncoder(state_dim, state_latent, hidden_dim=state_hidden, dropout=dropout)
        self.note_encoder = MLPEncoder(note_dim, note_latent, hidden_dim=note_hidden, dropout=dropout)

        fusion_in = state_latent + note_latent
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(self, state_x, note_x):
        z_s = self.state_encoder(state_x)
        z_n = self.note_encoder(note_x)
        z = torch.cat([z_s, z_n], dim=1)
        return self.fusion_head(z)


# ============================================================
# Training helpers
# ============================================================
def _predict_proba_mlp(model: nn.Module, X: np.ndarray, device: str = "cuda", batch_size: int = 2048) -> np.ndarray:
    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list)


def train_state_only_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_eval: np.ndarray | None, y_eval: np.ndarray | None,
    device: str,
    hidden_dim: int = 128,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 60,
    seed: int = 42,
    log_prefix: str = "[State-MLP]",
):
    # _set_seed(seed)
    n_actions = int(np.max(y_train)) + 1

    model = SimpleMLP(in_dim=X_train.shape[1], n_actions=n_actions, hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).long()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True, drop_last=False)

    if X_eval is not None and y_eval is not None:
        Xev = torch.from_numpy(X_eval).float().to(device)
        yev_np = y_eval
    else:
        Xev = None
        yev_np = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(X_train)

        # epoch eval
        model.eval()
        with torch.no_grad():
            tr_logits = model(torch.from_numpy(X_train).float().to(device))
            tr_pred = torch.argmax(tr_logits, dim=1).cpu().numpy()
        tr_acc = accuracy_score(y_train, tr_pred)

        if Xev is not None:
            with torch.no_grad():
                ev_logits = model(Xev)
                ev_pred = torch.argmax(ev_logits, dim=1).cpu().numpy()
            ev_acc = accuracy_score(yev_np, ev_pred)
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f} | eval_acc={ev_acc:.4f}")
        else:
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f}")

    return model, n_actions


def train_note_only(
    Xn_train: np.ndarray, y_train: np.ndarray,
    Xn_eval: np.ndarray | None, y_eval: np.ndarray | None,
    device: str,
    note_hidden: int = 256,
    note_latent: int = 128,
    head_hidden: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 60,
    seed: int = 42,
    log_prefix: str = "[Note-Only]",
):
    # _set_seed(seed)
    n_actions = int(np.max(y_train)) + 1

    model = NoteOnlyPolicy(
        note_dim=int(Xn_train.shape[1]),
        n_actions=n_actions,
        note_hidden=note_hidden,
        note_latent=note_latent,
        head_hidden=head_hidden,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xn_tr = torch.from_numpy(Xn_train).float()
    y_tr = torch.from_numpy(y_train).long()
    loader = DataLoader(TensorDataset(Xn_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)

    if Xn_eval is not None and y_eval is not None:
        Xn_ev = torch.from_numpy(Xn_eval).float().to(device)
        y_ev_np = y_eval
    else:
        Xn_ev = None
        y_ev_np = None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for xn_b, yb in loader:
            xn_b = xn_b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xn_b)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xn_b.size(0)

        avg_loss = total_loss / len(Xn_train)

        # epoch eval
        model.eval()
        with torch.no_grad():
            tr_logits = model(torch.from_numpy(Xn_train).float().to(device))
            tr_pred = torch.argmax(tr_logits, dim=1).cpu().numpy()
        tr_acc = accuracy_score(y_train, tr_pred)

        if Xn_ev is not None:
            with torch.no_grad():
                ev_logits = model(Xn_ev)
                ev_pred = torch.argmax(ev_logits, dim=1).cpu().numpy()
            ev_acc = accuracy_score(y_ev_np, ev_pred)
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f} | eval_acc={ev_acc:.4f}")
        else:
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f}")

    return model, n_actions


def train_late_fusion(
    Xs_train: np.ndarray, Xn_train: np.ndarray, y_train: np.ndarray,
    Xs_eval: np.ndarray | None, Xn_eval: np.ndarray | None, y_eval: np.ndarray | None,
    device: str,
    state_hidden: int = 128,
    note_hidden: int = 256,
    state_latent: int = 64,
    note_latent: int = 128,
    head_hidden: int = 256,
    dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 60,
    seed: int = 42,
    log_prefix: str = "[Late-Fusion]",
):
    # _set_seed(seed)
    n_actions = int(np.max(y_train)) + 1

    model = LateFusionPolicy(
        state_dim=int(Xs_train.shape[1]),
        note_dim=int(Xn_train.shape[1]),
        n_actions=n_actions,
        state_hidden=state_hidden,
        note_hidden=note_hidden,
        state_latent=state_latent,
        note_latent=note_latent,
        head_hidden=head_hidden,
        dropout=dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xs_tr = torch.from_numpy(Xs_train).float()
    Xn_tr = torch.from_numpy(Xn_train).float()
    y_tr = torch.from_numpy(y_train).long()
    loader = DataLoader(TensorDataset(Xs_tr, Xn_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)

    if Xs_eval is not None and Xn_eval is not None and y_eval is not None:
        Xs_ev = torch.from_numpy(Xs_eval).float().to(device)
        Xn_ev = torch.from_numpy(Xn_eval).float().to(device)
        y_ev_np = y_eval
    else:
        Xs_ev, Xn_ev, y_ev_np = None, None, None

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for xs_b, xn_b, yb in loader:
            xs_b = xs_b.to(device, non_blocking=True)
            xn_b = xn_b.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xs_b, xn_b)
            loss = F.cross_entropy(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_n += bs

        avg_loss = total_loss / max(total_n, 1)

        # epoch eval
        model.eval()
        with torch.no_grad():
            tr_logits = model(
                torch.from_numpy(Xs_train).float().to(device),
                torch.from_numpy(Xn_train).float().to(device),
            )
            tr_pred = torch.argmax(tr_logits, dim=1).cpu().numpy()
        tr_acc = accuracy_score(y_train, tr_pred)

        if Xs_ev is not None:
            with torch.no_grad():
                ev_logits = model(Xs_ev, Xn_ev)
                ev_pred = torch.argmax(ev_logits, dim=1).cpu().numpy()
            ev_acc = accuracy_score(y_ev_np, ev_pred)
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f} | eval_acc={ev_acc:.4f}")
        else:
            print(f"{log_prefix} Epoch {ep:02d}/{epochs} | loss={avg_loss:.4f} | train_acc={tr_acc:.4f}")

    return model, n_actions


def predict_proba_late_fusion(model: nn.Module, Xs: np.ndarray, Xn: np.ndarray, device: str = "cuda", batch_size: int = 2048):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(Xs), batch_size):
            xs_b = torch.from_numpy(Xs[i:i+batch_size]).float().to(device)
            xn_b = torch.from_numpy(Xn[i:i+batch_size]).float().to(device)
            logits = model(xs_b, xn_b)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()
            probs_list.append(probs)
    return np.vstack(probs_list)


# ============================================================
# Train (RF + state-only + note-only + late-fusion)
# ============================================================
def train(model_type: str, target_data: str):
    train_buffer = ReplayBuffer(
        state_dim=43,
        embed_dim=4096,
        batch_size=128,
        target_data=target_data,
        buffer_path=f"./dataset/{target_data}/buffer_llama/",
        note_form=''
    ).load_original_dataset(only_test_set=False)

    test_buffer = ReplayBuffer(
        state_dim=43,
        embed_dim=4096,
        batch_size=128,
        target_data=target_data,
        buffer_path=f"./dataset/{target_data}/buffer_llama/",
        note_form=''
    ).load_original_dataset(only_test_set=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(PTH_DIR, exist_ok=True)

    # state/action
    Xs_train = torch.FloatTensor(train_buffer.state[:train_buffer.crt_size]).cpu().numpy()
    y_train = torch.LongTensor(train_buffer.action[:train_buffer.crt_size]).squeeze(-1).cpu().numpy()
    Xs_test  = torch.FloatTensor(test_buffer.state[:test_buffer.crt_size]).cpu().numpy()
    y_test  = torch.LongTensor(test_buffer.action[:test_buffer.crt_size]).squeeze(-1).cpu().numpy()

    # note embedding
    Xn_train = None
    Xn_test = None

    if model_type == "rf":
        rf = RandomForestClassifier(n_estimators=50, oob_score=True, random_state=42)
        rf.fit(Xs_train, y_train)
        print(f"[RF] Train Accuracy: {accuracy_score(y_train, rf.predict(Xs_train)):.4f}")
        print(f"[RF] Test  Accuracy: {accuracy_score(y_test,  rf.predict(Xs_test)):.4f}")

        rf_path = f"{PTH_DIR}/BC_{target_data}_rf.pkl"
        joblib.dump(rf, rf_path)
        print(f"[RF] saved: {rf_path}")

    elif model_type == "state_nn":
        model, n_actions = train_state_only_mlp(
            X_train=Xs_train, y_train=y_train,
            X_eval=Xs_test, y_eval=y_test,
            device=device,
            hidden_dim=128,
            dropout=0.1,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            epochs=60,
            seed=42,
            log_prefix="[State-MLP]",
        )

        ckpt_path = f"{PTH_DIR}/BC_{target_data}_state_nn.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": int(Xs_train.shape[1]),
                "n_actions": int(n_actions),
                "hidden_dim": int(128),
                "dropout": float(0.1),
                "model_type": "state_nn",
            },
            ckpt_path
        )
        print(f"[State-MLP] saved: {ckpt_path}")

    elif model_type == "note_only":
        Xn_train = _get_note_embeddings_from_buffer(train_buffer)
        Xn_test  = _get_note_embeddings_from_buffer(test_buffer)
        print(f"[Note-Only] note train/test: {Xn_train.shape} / {Xn_test.shape}")

        model, n_actions = train_note_only(
            Xn_train=Xn_train, y_train=y_train,
            Xn_eval=Xn_test, y_eval=y_test,
            device=device,
            note_hidden=512,
            note_latent=256,
            head_hidden=128,
            dropout=0.1,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            epochs=60,
            seed=42,
            log_prefix="[Note-Only]",
        )

        ckpt_path = f"{PTH_DIR}/BC_{target_data}_note_only_llama_5.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "note_dim": int(Xn_train.shape[1]),
                "n_actions": int(n_actions),
                "note_hidden": 512,
                "note_latent": 256,
                "head_hidden": 128,
                "dropout": 0.1,
                "model_type": "note_only",
            },
            ckpt_path
        )
        print(f"[Note-Only] saved: {ckpt_path}")

    elif model_type == "late_fusion":
        Xn_train = _get_note_embeddings_from_buffer(train_buffer)
        Xn_test  = _get_note_embeddings_from_buffer(test_buffer)
        print(f"[Late-Fusion] state train/test: {Xs_train.shape} / {Xs_test.shape}")
        print(f"[Late-Fusion] note  train/test: {Xn_train.shape} / {Xn_test.shape}")

        model, n_actions = train_late_fusion(
            Xs_train=Xs_train, Xn_train=Xn_train, y_train=y_train,
            Xs_eval=Xs_test, Xn_eval=Xn_test, y_eval=y_test,
            device=device,
            state_hidden=256,
            note_hidden=512,
            state_latent=128,
            note_latent=256,
            head_hidden=128,
            dropout=0.1,
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=256,
            epochs=60,
            seed=42,
            log_prefix="[Late-Fusion]",
        )

        ckpt_path = f"{PTH_DIR}/BC_{target_data}_late_fusion_llama_5.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "state_dim": int(Xs_train.shape[1]),
                "note_dim": int(Xn_train.shape[1]),
                "n_actions": int(n_actions),
                "state_hidden": 256,
                "note_hidden": 512,
                "state_latent": 128,
                "note_latent": 256,
                "head_hidden": 128,
                "dropout": 0.1,
                "model_type": "late_fusion",
            },
            ckpt_path
        )
        print(f"[Late-Fusion] saved: {ckpt_path}")

    else:
        raise ValueError("model_type must be one of: ['rf', 'state_nn', 'note_only', 'late_fusion']")


# ============================================================
# Inference
# ============================================================
def inference(
    model_type: str,
    target_data: str = "mimic4",
    flag: str = "scaled_snuh_test",
    buffer_path: str | None = None,
):
    if buffer_path is None:
        buffer_path = f"./dataset/{target_data}/buffer_no_summer_clinical_bert/"

    action = np.load(f"{buffer_path}{flag}_action.npy").squeeze().astype(int)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type in ("state_nn", "late_fusion", "rf"):
        state = np.load(f"{buffer_path}{flag}_state.npy")
    else:
        state = None

    if model_type in ("note_only", "late_fusion"):
        cand = [
            f"{buffer_path}{flag}_note.npy",
            f"{buffer_path}{flag}_note_emb.npy",
            f"{buffer_path}{flag}_clinicalbert.npy",
            f"{buffer_path}{flag}_note_embedding.npy",
        ]
        note_file = _load_first_existing(cand)
        if note_file is None:
            raise FileNotFoundError(f"note embedding 파일을 찾지 못했습니다. candidates={cand}")

        note_emb = np.load(note_file, allow_pickle=True)
        note_emb = _to_2d_float_array(note_emb)
    else:
        note_emb = None

    # ---- RF ----
    if model_type == "rf":
        rf_file = _load_first_existing([
            f"{PTH_DIR}/BC_{target_data}_rf.pkl",
            f"{PTH_DIR}/BC_{target_data}.pkl",
        ])
        if rf_file is None:
            raise FileNotFoundError("RF x.")
        rf = joblib.load(rf_file)
        proba_all = rf.predict_proba(state)

    # ---- state_nn ----
    elif model_type == "state_nn":
        ckpt_file = _load_first_existing([f"{PTH_DIR}/BC_{target_data}_state_nn.pt"])
        if ckpt_file is None:
            raise FileNotFoundError("state_nn ckpt x")

        ckpt = torch.load(ckpt_file, map_location=device)
        model = SimpleMLP(
            in_dim=ckpt["in_dim"],
            n_actions=ckpt["n_actions"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        proba_all = _predict_proba_mlp(model, state, device=device)

    # ---- note_only ----
    elif model_type == "note_only":
        ckpt_file = _load_first_existing([f"{PTH_DIR}/BC_{target_data}_note_only.pt"])
        if ckpt_file is None:
            raise FileNotFoundError("note_only ckpt x")

        ckpt = torch.load(ckpt_file, map_location=device)
        model = NoteOnlyPolicy(
            note_dim=ckpt["note_dim"],
            n_actions=ckpt["n_actions"],
            note_hidden=ckpt["note_hidden"],
            note_latent=ckpt["note_latent"],
            head_hidden=ckpt["head_hidden"],
            dropout=ckpt["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        proba_all = _predict_proba_mlp(model, note_emb, device=device)

    # ---- late_fusion ----
    elif model_type == "late_fusion":
        ckpt_file = _load_first_existing([f"{PTH_DIR}/BC_{target_data}_late_fusion.pt"])
        if ckpt_file is None:
            raise FileNotFoundError("late_fusion ckpt x")

        ckpt = torch.load(ckpt_file, map_location=device)
        model = LateFusionPolicy(
            state_dim=ckpt["state_dim"],
            note_dim=ckpt["note_dim"],
            n_actions=ckpt["n_actions"],
            state_hidden=ckpt["state_hidden"],
            note_hidden=ckpt["note_hidden"],
            state_latent=ckpt["state_latent"],
            note_latent=ckpt["note_latent"],
            head_hidden=ckpt["head_hidden"],
            dropout=ckpt["dropout"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        proba_all = predict_proba_late_fusion(model, state, note_emb, device=device)

    else:
        raise ValueError("model_type must be one of: ['rf', 'state_nn', 'note_only', 'late_fusion']")

    b_prob_np = proba_all[np.arange(action.shape[0]), action]
    out_path = f"{buffer_path}{flag}_BC_prob_{model_type}.npy"
    np.save(out_path, b_prob_np.reshape(-1, 1))
    print(f"saved: {out_path}")



def inference_ope():
    target_data = 'mimic3'
    flag = 'train_val'
    buffer_path = f"./dataset/{target_data}/buffer_clinical_bert_radiology/"

    action = np.load(f"{buffer_path}{flag}_action.npy").squeeze().astype(int)
    state = np.load(f"{buffer_path}{flag}_state.npy")
    print(np.load(f"{buffer_path}{flag}_action.npy").shape, state.shape)
    bc_clf = joblib.load(f"./pth/BC_{target_data}.pkl")
    proba_all   = bc_clf.predict_proba(state)  # (N, n_actions)
    b_prob_np   = proba_all[np.arange(action.shape[0]), action]  

    np.save(f"{buffer_path}{flag}_BC_prob.npy", b_prob_np.reshape(-1,1))


if __name__ == "__main__":
    # train(model_type="note_only", target_data="mimic3")
    # train(model_type="late_fusion", target_data="mimic3")
    # train(model_type="state_nn", target_data="mimic3")
    # train(model_type="rf", target_data="mimic3")

    # train(model_type="note_only", target_data="snuh")
    # train(model_type="late_fusion", target_data="snuh")

    inference(model_type="note_only", target_data="mimic4", flag="scaled_snuh_test")
    inference(model_type="late_fusion", target_data="mimic4", flag="scaled_snuh_test")
    # inference_ope()
