import os
import re
import json
import math
import time
import copy
import random
import argparse
import importlib.util
from types import SimpleNamespace
from typing import List, Optional, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Project modules (same folder)
from network import CQLContextGatedFusionMixerNet
from metric import (
    eval_wis_ci,
    eval_multi_step_doubly_robust_ci,
    eval_fqe_ci,
    eval_opera_ci,
)

try:
    from transformers import AutoTokenizer, AutoModel
except Exception as e:
    raise ImportError(
        "transformers install"
    ) from e

try:
    from peft import LoraConfig, get_peft_model
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _np_load(path: str, allow_pickle: bool = False) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=allow_pickle)


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    s = str(x)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_py_variable_list(py_path: str) -> List[str]:
    """
    Load a python file that defines a variable containing a list/array of note texts.
    Heuristics:
      - Prefer variable names in ['note', 'notes', 'data', 'texts', 'text']
      - Else pick the first list/tuple/np.ndarray found in module globals
    """
    spec = importlib.util.spec_from_file_location("_note_module", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    cand_names = ["note", "notes", "data", "texts", "text"]
    for name in cand_names:
        if hasattr(mod, name):
            obj = getattr(mod, name)
            return _coerce_to_text_list(obj, f"{py_path}:{name}")

    for k, v in vars(mod).items():
        if k.startswith("__"):
            continue
        if isinstance(v, (list, tuple, np.ndarray)):
            return _coerce_to_text_list(v, f"{py_path}:{k}")

    raise ValueError(f"No list-like variable found in {py_path}")


def _coerce_to_text_list(obj: Any, where: str) -> List[str]:
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, tuple):
        obj = list(obj)
    if not isinstance(obj, list):
        raise TypeError(f"{where} is not list-like. got={type(obj)}")
    return [_safe_text(x) for x in obj]


def _try_load_texts(base_dir: str, candidates: List[str]) -> Optional[List[str]]:
    """
    Try to load raw texts from candidate filenames under base_dir.
    Supports .py and .npy
    """
    for fname in candidates:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            continue
        if path.endswith(".py"):
            return _load_py_variable_list(path)
        if path.endswith(".npy"):
            arr = _np_load(path, allow_pickle=True)
            return _coerce_to_text_list(arr, path)
    return None


def _shift_next_texts(texts: List[str]) -> List[str]:
    if len(texts) == 0:
        return []
    return texts[1:] + [""]


def _build_context_from_episode_start(note_texts: List[str], done: np.ndarray) -> List[str]:
    """
    If explicit context/background notes do not exist, build context as the episode-start note
    and forward-fill it within the episode.
    """
    done_ = done.reshape(-1) if done.ndim > 1 else done
    N = len(note_texts)
    if len(done_) != N:
        raise ValueError(f"done length mismatch: len(done)={len(done_)}, len(note_texts)={N}")

    ctx = [""] * N
    start_idx = 0
    current_ctx = _safe_text(note_texts[0]) if N > 0 else ""
    for i in range(N):
        if i == start_idx:
            current_ctx = _safe_text(note_texts[i])
        ctx[i] = current_ctx
        if done_[i] > 0.5:
            start_idx = i + 1
            if start_idx < N:
                current_ctx = _safe_text(note_texts[start_idx])
    return ctx


def _cosine_warmup_lr_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _parse_torch_dtype(dtype_str: str, device: str) -> torch.dtype:
    s = (dtype_str or "").lower().strip()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32"):
        return torch.float32
    # auto
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


# -----------------------------
# Buffer (raw text)
# -----------------------------
class RawTextReplayBuffer:
    def __init__(
        self,
        buffer_dir: str,
        flag: str,
        note_form: str = "",
        device: str = "cuda",
    ):
        self.buffer_dir = buffer_dir
        self.flag = flag
        self.note_form = note_form
        self.device = device

        self.state = _np_load(os.path.join(buffer_dir, f"{flag}_state.npy"))
        self.next_state = _np_load(os.path.join(buffer_dir, f"{flag}_next_state.npy"))
        self.action = _np_load(os.path.join(buffer_dir, f"{flag}_action.npy")).reshape(-1, 1).astype(np.int64)
        self.reward = _np_load(os.path.join(buffer_dir, f"{flag}_reward.npy")).reshape(-1, 1).astype(np.float32)
        self.done = _np_load(os.path.join(buffer_dir, f"{flag}_done.npy")).reshape(-1, 1).astype(np.float32)

        self.crt_size = int(self.state.shape[0])

        bc_path = os.path.join(buffer_dir, f"{flag}_BC_prob.npy")
        if os.path.exists(bc_path):
            self.bc_prob = _np_load(bc_path).reshape(-1, 1).astype(np.float32)
        else:
            self.bc_prob = None

        note_candidates = [
            f"{flag}{note_form}_note.py",
            f"{flag}{note_form}_note.npy",
            f"{flag}_note.py",
            f"{flag}_note.npy",
        ]
        note_texts = _try_load_texts(buffer_dir, note_candidates)
        if note_texts is None:
            raise FileNotFoundError(
                "Raw note file not found. Tried: "
                + ", ".join([os.path.join(buffer_dir, c) for c in note_candidates])
            )
        self.note_text = note_texts

        next_note_candidates = [
            f"{flag}{note_form}_next_note.py",
            f"{flag}{note_form}_next_note.npy",
            f"{flag}_next_note.py",
            f"{flag}_next_note.npy",
        ]
        next_note_texts = _try_load_texts(buffer_dir, next_note_candidates)
        if next_note_texts is None:
            next_note_texts = _shift_next_texts(self.note_text)
        self.next_note_text = next_note_texts

        ctx_candidates = [
            f"{flag}{note_form}_context_note.py",
            f"{flag}{note_form}_context_note.npy",
            f"{flag}_context_note.py",
            f"{flag}_context_note.npy",
            f"{flag}{note_form}_note_bg_only.py",
            f"{flag}{note_form}_note_bg_only.npy",
            f"{flag}_note_bg_only.py",
            f"{flag}_note_bg_only.npy",
        ]
        ctx_texts = _try_load_texts(buffer_dir, ctx_candidates)
        if ctx_texts is None:
            ctx_texts = _build_context_from_episode_start(self.note_text, self.done)
        self.context_text = ctx_texts

        next_ctx_candidates = [
            f"{flag}{note_form}_next_context_note.py",
            f"{flag}{note_form}_next_context_note.npy",
            f"{flag}_next_context_note.py",
            f"{flag}_next_context_note.npy",
            f"{flag}{note_form}_next_note_bg_only.py",
            f"{flag}{note_form}_next_note_bg_only.npy",
            f"{flag}_next_note_bg_only.py",
            f"{flag}_next_note_bg_only.npy",
        ]
        next_ctx_texts = _try_load_texts(buffer_dir, next_ctx_candidates)
        if next_ctx_texts is None:
            next_ctx_texts = _shift_next_texts(self.context_text)
        self.next_context_text = next_ctx_texts

        self._check_lengths()

    def _check_lengths(self) -> None:
        N = self.crt_size
        for name, arr in [
            ("next_state", self.next_state),
            ("action", self.action),
            ("reward", self.reward),
            ("done", self.done),
        ]:
            if arr.shape[0] != N:
                raise ValueError(f"{name} length mismatch: {arr.shape[0]} vs {N}")

        for name, lst in [
            ("note_text", self.note_text),
            ("next_note_text", self.next_note_text),
            ("context_text", self.context_text),
            ("next_context_text", self.next_context_text),
        ]:
            if len(lst) != N:
                raise ValueError(f"{name} length mismatch: {len(lst)} vs {N}")

        if self.bc_prob is not None and self.bc_prob.shape[0] != N:
            raise ValueError(f"bc_prob length mismatch: {self.bc_prob.shape[0]} vs {N}")

    def sample_indices(self, batch_size: int) -> np.ndarray:
        return np.random.randint(0, self.crt_size, size=batch_size)

    def get_batch(self, idx: np.ndarray) -> Dict[str, Any]:
        batch = {
            "state": torch.as_tensor(self.state[idx], dtype=torch.float32, device=self.device),
            "next_state": torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=self.device),
            "action": torch.as_tensor(self.action[idx], dtype=torch.long, device=self.device),
            "reward": torch.as_tensor(self.reward[idx], dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=self.device),
        }
        batch["note_text"] = [self.note_text[i] for i in idx.tolist()]
        batch["ctx_text"] = [self.context_text[i] for i in idx.tolist()]
        batch["next_note_text"] = [self.next_note_text[i] for i in idx.tolist()]
        batch["next_ctx_text"] = [self.next_context_text[i] for i in idx.tolist()]

        if self.bc_prob is not None:
            batch["bc_prob"] = torch.as_tensor(self.bc_prob[idx], dtype=torch.float32, device=self.device)
        else:
            batch["bc_prob"] = None
        return batch


# -----------------------------
# Llama Text Encoder (LoRA optional)
# -----------------------------
class LlamaTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        pooling: str = "last",  # "last" | "mean"
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
        tokenizer_use_fast: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = int(max_length)
        self.pooling = pooling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=bool(tokenizer_use_fast))

        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        self.llm = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        if len(self.tokenizer) != self.llm.get_input_embeddings().num_embeddings:
            self.llm.resize_token_embeddings(len(self.tokenizer))

        if gradient_checkpointing and hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable()

        self.hidden_size = int(getattr(self.llm.config, "hidden_size", 4096))

        self.is_peft = False
        if use_lora:
            if not _HAS_PEFT:
                raise ImportError(
                    "pip install peft"
                )
            if lora_target_modules is None:
                # Llama 계열 기본 타깃
                lora_target_modules = ["q_proj", "v_proj"]

            cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                target_modules=lora_target_modules,
                task_type="FEATURE_EXTRACTION",
            )
            self.llm = get_peft_model(self.llm, cfg)
            self.is_peft = True

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Returns: (B, hidden_size) embeddings
        """
        texts = [_safe_text(t) for t in texts]
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(self.llm.device) for k, v in tok.items()}

        out = self.llm(**tok)
        last = out.last_hidden_state  # (B, L, H)
        attn = tok.get("attention_mask", None)

        if self.pooling == "mean":
            if attn is None:
                return last.mean(dim=1)
            mask = attn.unsqueeze(-1).float()  # (B,L,1)
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return summed / denom

        if attn is None:
            return last[:, -1, :]
        idx_last = attn.sum(dim=1).clamp_min(1) - 1  # (B,)
        idx_last = idx_last.view(-1, 1, 1).expand(-1, 1, last.size(-1))  # (B,1,H)
        pooled = last.gather(dim=1, index=idx_last).squeeze(1)  # (B,H)
        return pooled

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.encode(texts)


# -----------------------------
# Training + Export + OPE
# -----------------------------
def train_one_epoch_stepwise(
    q_net: nn.Module,
    q_target: nn.Module,
    encoder: LlamaTextEncoder,
    buffer: RawTextReplayBuffer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: str,
    steps: int,
    batch_size: int,
    discount: float,
    cql_alpha: float,
    target_update_freq: int,
    grad_clip: float,
    use_amp: bool,
    log_interval: int = 10,
) -> None:
    q_net.train()
    encoder.train()

    for it in range(1, steps + 1):
        idx = buffer.sample_indices(batch_size)
        batch = buffer.get_batch(idx)

        state = batch["state"]
        next_state = batch["next_state"]
        action = batch["action"]
        reward = batch["reward"]
        done = batch["done"]

        note_text = batch["note_text"]
        ctx_text = batch["ctx_text"]
        next_note_text = batch["next_note_text"]
        next_ctx_text = batch["next_ctx_text"]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # current embeddings: allow grads to flow into encoder
            note_emb = encoder(note_text)          # (B,H)
            ctx_emb = encoder(ctx_text)            # (B,H)

            # next embeddings: target computation, no grad
            with torch.no_grad():
                next_note_emb = encoder.encode(next_note_text)
                next_ctx_emb = encoder.encode(next_ctx_text)

                q_next = q_target(next_state, next_note_emb, next_ctx_emb)
                next_act = q_next.argmax(dim=1, keepdim=True)
                target_q = reward + (1.0 - done) * discount * q_next.gather(1, next_act)

            q_pred = q_net(state, note_emb, ctx_emb)
            current_q = q_pred.gather(1, action)

            td_loss = F.mse_loss(current_q, target_q)

            alpha = float(cql_alpha)
            lse = torch.logsumexp(q_pred / alpha, dim=1, keepdim=True) * alpha
            cql_loss = lse.mean() - current_q.mean()

            loss = td_loss + cql_loss

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(q_net.parameters()) + list(encoder.parameters()), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if it % target_update_freq == 0:
            q_target.load_state_dict(q_net.state_dict())

        if log_interval > 0 and it % log_interval == 0:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            print(
                f"[train] it={it:>7d} | loss={loss.item():.4f} | td={td_loss.item():.4f} | cql={cql_loss.item():.4f} | lr={lr:.2e}",
                flush=True,
            )


@torch.no_grad()
def encode_full_buffer(
    encoder: LlamaTextEncoder,
    buffer: RawTextReplayBuffer,
    batch_size: int = 64,
    device: str = "cuda",
) -> SimpleNamespace:
    """
    Encode the entire buffer into float embeddings for note/context and next_note/next_context.
    Returns a lightweight object compatible with metric.py expectations.
    """
    encoder.eval()

    N = buffer.crt_size
    H = encoder.hidden_size

    def _alloc():
        return np.zeros((N, H), dtype=np.float32)

    note = _alloc()
    next_note = _alloc()
    ctx = _alloc()
    next_ctx = _alloc()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        nt = buffer.note_text[start:end]
        nnt = buffer.next_note_text[start:end]
        ct = buffer.context_text[start:end]
        nct = buffer.next_context_text[start:end]

        emb_nt = encoder.encode(nt).to("cpu").float().numpy()
        emb_nnt = encoder.encode(nnt).to("cpu").float().numpy()
        emb_ct = encoder.encode(ct).to("cpu").float().numpy()
        emb_nct = encoder.encode(nct).to("cpu").float().numpy()

        note[start:end] = emb_nt
        next_note[start:end] = emb_nnt
        ctx[start:end] = emb_ct
        next_ctx[start:end] = emb_nct

    out = SimpleNamespace()
    out.crt_size = int(N)

    out.state = buffer.state.astype(np.float32)
    out.next_state = buffer.next_state.astype(np.float32)
    out.action = buffer.action.astype(np.int64)
    out.reward = buffer.reward.astype(np.float32)
    out.done = buffer.done.astype(np.float32)

    out.note = note
    out.next_note = next_note
    out.note_bg_only = ctx
    out.next_note_bg_only = next_ctx

    if buffer.bc_prob is not None:
        out.bc_prob = buffer.bc_prob.astype(np.float32)
    else:
        n_actions = int(np.max(buffer.action)) + 1
        out.bc_prob = np.full((N, 1), 1.0 / float(n_actions), dtype=np.float32)

    return out


def run_ope(
    algorithm: str,
    q_net: nn.Module,
    q_target: nn.Module,
    encoded_buffer,
    device: str,
    gamma: float,
    batch_size: int,
    clip: float,
    n_bootstrap: int,
    alpha_ci: float,
) -> Dict[str, Any]:

    policy = SimpleNamespace()
    policy.Q = q_net.to(device).eval()
    policy.Q_target = q_target.to(device).eval()
    policy.device = device

    results: Dict[str, Any] = {}

    wis_mean, wis_lo, wis_hi = eval_wis_ci(
        algorithm=algorithm,
        policy=policy,
        replay_buffer=encoded_buffer,
        clip=clip,
        gamma=gamma,
        device=device,
        n_bootstrap=n_bootstrap,
        alpha=alpha_ci,
    )
    results["WIS"] = {"mean": wis_mean, "ci_lower": wis_lo, "ci_upper": wis_hi}

    dr_mean, dr_lo, dr_hi = eval_multi_step_doubly_robust_ci(
        algorithm=algorithm,
        policy=policy,
        replay_buffer=encoded_buffer,
        clip=clip,
        batch_size=batch_size,
        gamma=gamma,
        device=device,
        n_bootstrap=n_bootstrap,
        alpha=alpha_ci,
    )
    results["DR"] = {"mean": dr_mean, "ci_lower": dr_lo, "ci_upper": dr_hi}

    fqe_mean, fqe_lo, fqe_hi = eval_fqe_ci(
        algorithm=algorithm,
        policy=policy,
        replay_buffer=encoded_buffer,
        gamma=gamma,
        device=device,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        alpha=alpha_ci,
    )
    results["FQE"] = {"mean": fqe_mean, "ci_lower": fqe_lo, "ci_upper": fqe_hi}

    opera_mean, opera_lo, opera_hi, alphas = eval_opera_ci(
        algorithm=algorithm,
        policy=policy,
        replay_buffer=encoded_buffer,
        gamma=gamma,
        device=device,
        n_bootstrap=n_bootstrap,
        alpha=alpha_ci,
    )
    results["OPERA"] = {
        "mean": opera_mean,
        "ci_lower": opera_lo,
        "ci_upper": opera_hi,
        "alphas": alphas.tolist() if hasattr(alphas, "tolist") else alphas,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    # paths
    parser.add_argument("--buffer_dir", type=str, required=True, help="e.g., ./dataset/mimic3/buffer_raw_text/")
    parser.add_argument("--train_flag", type=str, default="train_val", help="prefix for training buffer files")
    parser.add_argument("--test_flag", type=str, default="test", help="prefix for test buffer files")
    parser.add_argument("--note_form", type=str, default="", help="suffix used in file naming, e.g., _raw / _impute / _stack")

    # text model (kept arg name for backward compatibility)
    parser.add_argument(
        "--bert_model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="text encoder",
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "cls", "pooler"])
    parser.add_argument("--text_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--tokenizer_use_fast", action="store_true")

    # LoRA
    parser.add_argument("--no_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,v_proj",
        help="comma-separated target modules for LoRA",
    )

    # CQL training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--cql_alpha", type=float, default=1.0)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")

    # optimizer + schedule
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # OPE
    parser.add_argument("--run_ope", action="store_true")
    parser.add_argument("--ope_gamma", type=float, default=0.98)
    parser.add_argument("--ope_batch_size", type=int, default=64)
    parser.add_argument("--ope_clip", type=float, default=5.0)
    parser.add_argument("--ope_bootstrap", type=int, default=1000)
    parser.add_argument("--ope_alpha", type=float, default=0.05)

    # outputs
    parser.add_argument("--out_dir", type=str, default="./pth_lora")
    parser.add_argument("--ckpt_name", type=str, default="llama31_8b_it_cql_lora.pt")
    parser.add_argument("--export_encoded_test", action="store_true", help="export encoded test embeddings to .npy")

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # pooling
    pooling = args.pooling
    if pooling in ("cls", "pooler"):
        pooling = "last"

    # dtype
    torch_dtype = _parse_torch_dtype(args.text_dtype, device=device)

    # 1) Load raw-text buffers
    train_buf = RawTextReplayBuffer(
        buffer_dir=args.buffer_dir,
        flag=args.train_flag,
        note_form=args.note_form,
        device=device,
    )
    test_buf = RawTextReplayBuffer(
        buffer_dir=args.buffer_dir,
        flag=args.test_flag,
        note_form=args.note_form,
        device=device,
    )

    state_dim = int(train_buf.state.shape[1])
    n_actions = int(np.max(train_buf.action)) + 1

    # 2) Build encoder (Llama + optional LoRA)
    lora_targets = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
    encoder = LlamaTextEncoder(
        model_name=args.bert_model,
        max_length=args.max_length,
        pooling=pooling,
        use_lora=(not args.no_lora),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets if len(lora_targets) > 0 else None,
        torch_dtype=torch_dtype,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        tokenizer_use_fast=bool(args.tokenizer_use_fast),
    ).to(device)

    note_emb_dim = int(encoder.hidden_size)

    # 3) Build Q networks
    q_net = CQLContextGatedFusionMixerNet(
        state_dim=state_dim,
        num_actions=n_actions,
        hidden_node=256,
        activation="relu",
        note_emb_dim=note_emb_dim,
    ).to(device)
    q_target = copy.deepcopy(q_net).to(device)

    # 4) Optimizer + cosine(warmup) scheduler
    params = list(q_net.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = int(args.train_steps)
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_warmup_lr_lambda(s, warmup_steps, total_steps)
    )

    scaler = GradScaler(enabled=bool(args.use_amp))

    # 5) Train
    print(
        f"[setup] device={device} | state_dim={state_dim} | n_actions={n_actions} | note_emb_dim={note_emb_dim} "
        f"| lora={'off' if args.no_lora else 'on'} | model={args.bert_model} | pooling={pooling} | dtype={torch_dtype}",
        flush=True,
    )
    t0 = time.time()
    train_one_epoch_stepwise(
        q_net=q_net,
        q_target=q_target,
        encoder=encoder,
        buffer=train_buf,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        steps=total_steps,
        batch_size=args.batch_size,
        discount=args.discount,
        cql_alpha=args.cql_alpha,
        target_update_freq=args.target_update_freq,
        grad_clip=args.grad_clip,
        use_amp=bool(args.use_amp),
        log_interval=10,
    )
    print(f"[train] done. elapsed={time.time() - t0:.1f}s", flush=True)

    # 6) Save checkpoint
    ckpt_path = os.path.join(args.out_dir, args.ckpt_name)
    ckpt = {
        "q_state_dict": q_net.state_dict(),
        "q_target_state_dict": q_target.state_dict(),
        "text_model": args.bert_model,
        "encoder_state_dict": encoder.state_dict(),
        "note_emb_dim": note_emb_dim,
        "state_dim": state_dim,
        "n_actions": n_actions,
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_path)
    print(f"[save] {ckpt_path}", flush=True)

    # 7) Encode test buffer and export + run OPE
    encoded_test = encode_full_buffer(
        encoder=encoder,
        buffer=test_buf,
        batch_size=args.ope_batch_size,
        device=device,
    )

    if args.export_encoded_test:
        export_dir = os.path.join(args.out_dir, "encoded_test")
        os.makedirs(export_dir, exist_ok=True)
        np.save(os.path.join(export_dir, f"{args.test_flag}_note_embedding.npy"), encoded_test.note)
        np.save(os.path.join(export_dir, f"{args.test_flag}_next_note_embedding.npy"), encoded_test.next_note)
        np.save(os.path.join(export_dir, f"{args.test_flag}_context_embedding.npy"), encoded_test.note_bg_only)
        np.save(os.path.join(export_dir, f"{args.test_flag}_next_context_embedding.npy"), encoded_test.next_note_bg_only)
        np.save(os.path.join(export_dir, f"{args.test_flag}_state.npy"), encoded_test.state)
        np.save(os.path.join(export_dir, f"{args.test_flag}_next_state.npy"), encoded_test.next_state)
        np.save(os.path.join(export_dir, f"{args.test_flag}_action.npy"), encoded_test.action)
        np.save(os.path.join(export_dir, f"{args.test_flag}_reward.npy"), encoded_test.reward)
        np.save(os.path.join(export_dir, f"{args.test_flag}_done.npy"), encoded_test.done)
        np.save(os.path.join(export_dir, f"{args.test_flag}_BC_prob.npy"), encoded_test.bc_prob)
        print(f"[export] encoded test saved to: {export_dir}", flush=True)

    if args.run_ope:
        algorithm = "CQL_Cross_Context_Attention"
        ope_res = run_ope(
            algorithm=algorithm,
            q_net=q_net,
            q_target=q_target,
            encoded_buffer=encoded_test,
            device=device,
            gamma=args.ope_gamma,
            batch_size=args.ope_batch_size,
            clip=args.ope_clip,
            n_bootstrap=args.ope_bootstrap,
            alpha_ci=args.ope_alpha,
        )
        out_json = os.path.join(args.out_dir, "ope_results_llama_max_length_1024.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(ope_res, f, indent=2)
        print(f"[ope] results saved: {out_json}", flush=True)
        print(json.dumps(ope_res, indent=2), flush=True)


if __name__ == "__main__":
    main()
