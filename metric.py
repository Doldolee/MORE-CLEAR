from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch
import copy
from tqdm import tqdm
import numpy as np
import cvxpy as cp

from util import *
from BC import *

def pred_q_value(algorithm, policy, state, note, note_bg_only):
    if algorithm == 'CQL_Cross_Attention':
        return policy.Q(state, note)
    elif algorithm == 'CQL':
        return policy.Q(state)
    elif algorithm == 'CQL_Text':
        return policy.Q(note)
    elif algorithm == 'CQL_Cross_Context_Attention':
        return policy.Q(state, note, note_bg_only)
    elif algorithm == 'IQL':
        q1 = policy.critic1(state)
        q2 = policy.critic2(state)
        min_q = torch.min(q1, q2)
        return min_q
    
def pred_action(algorithm, policy, state, note, note_bg_only):
    if algorithm =='CQL_Cross_Attention':
        return policy.action(state, note)
    elif algorithm == 'CQL':
        return policy.action(state)
    elif algorithm == 'CQL_Text':
        return policy.action(note)
    elif algorithm == 'CQL_Cross_Context_Attention':
        return policy.action(state, note, note_bg_only)
    elif algorithm == 'IQL':
        return policy.action(state)

def collect_bellman_residuals(
    algorithm,
    policy,
    replay_buffer,
    gamma: float = 0.98,
    device: str = "cuda",
    batch_size: int = 256,
):

    if algorithm == 'IQL':
        policy.actor.to(device);   policy.actor.eval()
    else:
        policy.Q.to(device);       policy.Q.eval()

    N = replay_buffer.crt_size
    notes       = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    next_notes  = torch.FloatTensor(replay_buffer.next_note[:N]).to(device)
    states      = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    next_states = torch.FloatTensor(replay_buffer.next_state[:N]).to(device)
    actions     = torch.LongTensor(replay_buffer.action[:N]).to(device)
    rewards     = torch.FloatTensor(replay_buffer.reward[:N]).to(device).squeeze(-1)
    dones       = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)
    note_bg_only       = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    next_note_bg_only  = torch.FloatTensor(replay_buffer.next_note_bg_only[:N]).to(device)

    dataset = torch.utils.data.TensorDataset(
        states, notes, actions, rewards, next_states, next_notes, dones, note_bg_only, next_note_bg_only
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    residuals = []
    with torch.no_grad():
        for state_b, note_b, action_b, reward_b, next_s_b, next_n_b, done_b, note_bg_only, next_note_bg_only in loader:
            # Q(s,a)
            q       = pred_q_value(algorithm, policy, state_b, note_b, note_bg_only)
            q_sa    = q.gather(1, action_b).squeeze(-1)

            # max_a' Q(s',a')
            q_next  = pred_q_value(algorithm, policy, next_s_b, next_n_b, note_bg_only)
            v_next  = q_next.max(dim=1)[0] * (1.0 - done_b)

            # Bellman residual
            delta   = reward_b + gamma * v_next - q_sa

            residuals.append(delta.cpu().numpy())
   
    return np.concatenate(residuals, axis=0)

import joblib
def eval_policy_survival_rate(algorithm,
                              policy,
                              replay_buffer,
                              tol: float,
                              device: str = "cuda",
                              alpha: float = 0.4,
                              beta: float = 0.6,
                              q: float = 90.0,
                              bottom_q: float = 10.0,
                              default_rate: float = 0.70,
                              n_bootstrap: int = 1000,
                              ci_alpha: float = 0.05,
                            ):

    N = replay_buffer.crt_size

    notes   = torch.tensor(replay_buffer.note[:N],   dtype=torch.float32, device=device)
    notes_bg_only   = torch.tensor(replay_buffer.note_bg_only[:N],   dtype=torch.float32, device=device)
    states  = torch.tensor(replay_buffer.state[:N],  dtype=torch.float32, device=device)
    acts    = torch.tensor(replay_buffer.action[:N], dtype=torch.int64,   device=device)
    rewards = torch.tensor(replay_buffer.reward[:N], dtype=torch.float32, device=device)
    dones   = torch.tensor(replay_buffer.done[:N],   dtype=torch.float32, device=device)

    if algorithm == 'IQL':
        policy.actor.eval()
    else:
        policy.Q.eval()

    ep_outcomes = []
    vaso_diffs  = []
    iv_diffs    = []

    ai_records, clin_records = [], []



    # ---------------------------
    # 1) state-only(SimpleMLP) 
    # ---------------------------
    ckpt_state = torch.load("./pth/BC_mimic3_nn_5.pt", map_location=device)
    bc_state = SimpleMLP(
        in_dim=ckpt_state["in_dim"],
        n_actions=ckpt_state["n_actions"],
        hidden_dim=ckpt_state["hidden_dim"],
        dropout=ckpt_state["dropout"],
    ).to(device)
    bc_state.load_state_dict(ckpt_state["state_dict"])
    bc_state.eval()

    # ---------------------------
    # 2) note-only 
    # ---------------------------
    ckpt_note = torch.load("./pth/BC_snuh_note_only_llama_5.pt", map_location=device)
    bc_note = NoteOnlyPolicy(
        note_dim=ckpt_note["note_dim"],
        n_actions=ckpt_note["n_actions"],
        note_hidden=ckpt_note.get("note_hidden", 512),
        note_latent=ckpt_note.get("note_latent", 256),
        head_hidden=ckpt_note.get("head_hidden", 128),
        dropout=ckpt_note.get("dropout", 0.1),
    ).to(device)
    bc_note.load_state_dict(ckpt_note["state_dict"])
    bc_note.eval()

    # ---------------------------
    # 3) late-fusion
    # ---------------------------
    ckpt_fusion = torch.load("./pth/BC_snuh_late_fusion_llama_5.pt", map_location=device)
    bc_fusion = LateFusionPolicy(
        state_dim=ckpt_fusion["state_dim"],
        note_dim=ckpt_fusion["note_dim"],
        n_actions=ckpt_fusion["n_actions"],
        state_hidden=ckpt_fusion.get("state_hidden", 256),
        note_hidden=ckpt_fusion.get("note_hidden", 512),
        state_latent=ckpt_fusion.get("state_latent", 128),
        note_latent=ckpt_fusion.get("note_latent", 256),
        head_hidden=ckpt_fusion.get("head_hidden", 128),
        dropout=ckpt_fusion.get("dropout", 0.1),
    ).to(device)
    bc_fusion.load_state_dict(ckpt_fusion["state_dict"])
    bc_fusion.eval()

    # ---------------------------
    # 4) using example
    # ---------------------------
    # bc_type = "rl" | "bc_state_nn" | "bc_note_only" | "bc_late_fusion"
    model_type = "rl"  
    note_input = notes
    for i in range(N):
        with torch.no_grad():
            if model_type == 'rl':
                pa = pred_action(algorithm, policy, states[i:i+1], notes[i:i+1], notes_bg_only[i:i+1])

            elif model_type == "bc_state_nn":
                pa = bc_state(states[i:i+1]).argmax(dim=1).item()

            elif model_type == "bc_note_only":
                pa = bc_note(note_input[i:i+1]).argmax(dim=1).item()

            elif model_type == "bc_late_fusion":
                pa = bc_fusion(states[i:i+1], note_input[i:i+1]).argmax(dim=1).item()

            else:
                raise ValueError("bc_type must be one of ['state_nn', 'note_only', 'late_fusion']")
    
        ai_act   = pa
        clin_act = acts[i].item()

        ai_records.append(ai_act)
        clin_records.append(clin_act)

        if dones[i].item() == 1.0:
            r = rewards[i].item()
            assert r in (-1, 1), "Reward must be -1 or 1"
            ep_outcomes.append((r + 1) / 2)

            cv = [(a % 5) + 1 for a in clin_records]
            av = [(a % 5) + 1 for a in ai_records]
            vaso_diffs.append(np.sum([abs(x - y) for x, y in zip(cv, av)]))

            ci = [a // 5 for a in clin_records]
            ai_ = [a // 5 for a in ai_records]
            iv_diffs.append(np.sum([abs(x - y) for x, y in zip(ci, ai_)]))

            ai_records.clear()
            clin_records.clear()

    ep_outcomes = np.array(ep_outcomes, dtype=float)
    vaso_diffs  = np.array(vaso_diffs,  dtype=float)
    iv_diffs    = np.array(iv_diffs,    dtype=float)
    comb_diffs  = alpha * iv_diffs + beta * vaso_diffs
    E = len(ep_outcomes)
    surv_overall_pt = float(np.mean(ep_outcomes)) if E > 0 else default_rate

    th_high = np.percentile(comb_diffs, q)
    mask_high = comb_diffs > th_high
    surv_comb_high_pt = float(np.mean(ep_outcomes[mask_high])) if mask_high.any() else default_rate

    th_low = np.percentile(comb_diffs, bottom_q)
    mask_low = comb_diffs < th_low
    surv_comb_low_pt = float(np.mean(ep_outcomes[mask_low])) if mask_low.any() else default_rate

    th_vaso_high = np.percentile(vaso_diffs, q)
    mask_vaso_high = vaso_diffs > th_vaso_high
    surv_vaso_pt = float(np.mean(ep_outcomes[mask_vaso_high])) if mask_vaso_high.any() else default_rate

    th_vaso_low = np.percentile(vaso_diffs, bottom_q)
    mask_vaso_low = vaso_diffs < th_vaso_low
    surv_vaso_low_pt = float(np.mean(ep_outcomes[mask_vaso_low])) if mask_vaso_low.any() else default_rate

    th_iv_high = np.percentile(iv_diffs, q)
    mask_iv_high = iv_diffs > th_iv_high
    surv_iv_pt = float(np.mean(ep_outcomes[mask_iv_high])) if mask_iv_high.any() else default_rate

    th_iv_low = np.percentile(iv_diffs, bottom_q)
    mask_iv_low = iv_diffs < th_iv_low
    surv_iv_low_pt = float(np.mean(ep_outcomes[mask_iv_low])) if mask_iv_low.any() else default_rate

    # E = len(ep_outcomes)
    idxs_base = np.arange(E)
    boot_comb_high, boot_comb_low = [], []
    boot_vaso_high, boot_vaso_low = [], []
    boot_iv_high, boot_iv_low = [], []
    boot_overall = [] 

    for _ in range(n_bootstrap):
        idxs = np.random.choice(idxs_base, size=E, replace=True)
        s_outs = ep_outcomes[idxs]
        s_comb = comb_diffs[idxs]
        s_vaso = vaso_diffs[idxs]
        s_iv   = iv_diffs[idxs]

        th_ch = np.percentile(s_comb, q)
        m_ch = s_comb > th_ch
        boot_comb_high.append(float(np.mean(s_outs[m_ch])) if m_ch.any() else default_rate)

        th_cl = np.percentile(s_comb, bottom_q)
        m_cl = s_comb < th_cl
        boot_comb_low.append(float(np.mean(s_outs[m_cl])) if m_cl.any() else default_rate)

        th_vh = np.percentile(s_vaso, q)
        m_vh = s_vaso > th_vh
        boot_vaso_high.append(float(np.mean(s_outs[m_vh])) if m_vh.any() else default_rate)

        th_vl = np.percentile(s_vaso, bottom_q)
        m_vl = s_vaso < th_vl
        boot_vaso_low.append(float(np.mean(s_outs[m_vl])) if m_vl.any() else default_rate)

        th_ih = np.percentile(s_iv, q)
        m_ih = s_iv > th_ih
        boot_iv_high.append(float(np.mean(s_outs[m_ih])) if m_ih.any() else default_rate)

        th_il = np.percentile(s_iv, bottom_q)
        m_il = s_iv < th_il
        boot_iv_low.append(float(np.mean(s_outs[m_il])) if m_il.any() else default_rate)

        boot_overall.append(float(np.mean(s_outs)) if len(s_outs) > 0 else default_rate)

    low_p, high_p = 100 * (ci_alpha/2), 100 * (1 - ci_alpha/2)
    comb_high_ci = (np.percentile(boot_comb_high, low_p), np.percentile(boot_comb_high, high_p))
    comb_low_ci  = (np.percentile(boot_comb_low,  low_p), np.percentile(boot_comb_low,  high_p))
    vaso_ci      = (np.percentile(boot_vaso_high, low_p), np.percentile(boot_vaso_high, high_p))
    vaso_low_ci  = (np.percentile(boot_vaso_low,  low_p), np.percentile(boot_vaso_low,  high_p))
    iv_ci        = (np.percentile(boot_iv_high,   low_p), np.percentile(boot_iv_high,   high_p))
    iv_low_ci    = (np.percentile(boot_iv_low,    low_p), np.percentile(boot_iv_low,    high_p))
    overall_ci = (np.percentile(boot_overall, low_p), np.percentile(boot_overall, high_p))
    print("baseline survival rate: ", surv_overall_pt, overall_ci)
    return (
        surv_comb_low_pt,  comb_low_ci,
        surv_comb_high_pt, comb_high_ci,
        surv_vaso_low_pt,  vaso_low_ci,
        surv_vaso_pt,      vaso_ci,
        surv_iv_low_pt,    iv_low_ci,
        surv_iv_pt,        iv_ci,
    )

def eval_fqe_ci(algorithm,
                policy,
                replay_buffer,
                gamma: float = 0.98,
                device: str = "cuda",
                num_epochs: int = 20,
                batch_size: int = 256,
                lr: float = 1e-3,
                target_update_freq: int = 1,
                tau: float = 1.0,
                n_bootstrap: int = 1000,
                alpha: float = 0.05):

    
    fqe_Q        = copy.deepcopy(policy)
    fqe_target_Q = copy.deepcopy(policy)
    if algorithm == 'IQL':
        fqe_Q.critic1.train()
        fqe_target_Q.critic1.eval()
    else:
        fqe_Q.Q.train()
        fqe_target_Q.Q.eval()

    optimizer = torch.optim.Adam(fqe_Q.Q.parameters() if algorithm != 'IQL' else fqe_Q.critic1.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    N = replay_buffer.crt_size
    note       = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    next_note  = torch.FloatTensor(replay_buffer.next_note[:N]).to(device)
    note_bg_only       = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    next_note_bg_only  = torch.FloatTensor(replay_buffer.next_note_bg_only[:N]).to(device)

    state      = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    next_state = torch.FloatTensor(replay_buffer.next_state[:N]).to(device)
    action     = torch.LongTensor(replay_buffer.action[:N]).to(device)
    reward     = torch.FloatTensor(replay_buffer.reward[:N]).to(device)
    done       = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)

    dataset = FQEDataset(note, next_note, state, next_state, action, reward, done, note_bg_only, next_note_bg_only)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs+1):
        for n, nn, s, ns, a, r, d, note_bg_only, next_note_bg_only in loader:
            q_pred = pred_q_value(algorithm, fqe_Q, s, n, note_bg_only)
            q_sa = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # q_next, _, _ = fqe_target_Q(ns, nn)
                q_next = pred_q_value(algorithm, fqe_target_Q, ns, nn, next_note_bg_only)
                pi_next      = F.softmax(q_next, dim=1)
                v_next       = (pi_next * q_next).sum(dim=1) * (1 - d)
                target       = r + gamma * v_next
            loss = criterion(q_sa, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % target_update_freq == 0:
            if tau == 1.0:
                if algorithm == 'IQL':
                    fqe_target_Q.critic1.load_state_dict(fqe_Q.critic1.state_dict())
                else:
                    fqe_target_Q.Q.load_state_dict(fqe_Q.Q.state_dict())
            else:
                for p_online, p_target in zip(fqe_Q.Q.parameters() if algorithm != 'IQL' else fqe_Q.critic1.parameters(), 
                                              fqe_target_Q.Q.parameters() if algorithm != 'IQL' else fqe_target_Q.critic1.parameters()):
                    p_target.data.mul_(1 - tau)
                    p_target.data.add_(tau * p_online.data)

    if algorithm == 'IQL':
        fqe_Q.critic1.eval()
    else:
        fqe_Q.Q.eval()

    with torch.no_grad():
        
        end_idxs   = torch.nonzero(done, as_tuple=True)[0]
        start_idxs = torch.cat([torch.tensor([0], device=device),
                                end_idxs[:-1] + 1])
        v0_list = []
        for idx in start_idxs.tolist():
            s0     = state[idx].unsqueeze(0)
            n0     = note[idx].unsqueeze(0)
            note_bg_only = note[idx].unsqueeze(0)
            # q0, _, _ = fqe_Q(s0, n0)
            q0 = pred_q_value(algorithm, fqe_Q, s0, n0, note_bg_only)
            pi0      = F.softmax(q0, dim=1)
            v0_list.append((pi0 * q0).sum(dim=1).item())

    v0_arr = np.array(v0_list, dtype=np.float64)
    fqe_mean = float(v0_arr.mean())
    M = len(v0_arr)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(v0_arr, size=M, replace=True)
        boot_means.append(sample.mean())
    # np.save("./plot/radiology_bootstrap_fqe.npy", boot_means)
    ci_lower = np.percentile(boot_means, 100 * (alpha/2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha/2))

    print(f"FQE estimate: {fqe_mean:.4f}")
    print(f"{100*(1-alpha):.1f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return fqe_mean, ci_lower, ci_upper




def eval_multi_step_doubly_robust_ci(algorithm,
                                    policy,
                                    replay_buffer,
                                    clip: float = 5,
                                    batch_size: int = 256,
                                    gamma: float = 0.98,
                                    device: str = "cuda",
                                    n_bootstrap: int = 1000,
                                    alpha: float = 0.05
                                ):


    if algorithm == 'IQL':
        policy.actor.to(device)
        policy.actor.eval()
    else:
        policy.Q.to(device)
        policy.Q.eval()


    N = replay_buffer.crt_size
    done_all_np = replay_buffer.done[:N].squeeze(-1)

    note       = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    next_note  = torch.FloatTensor(replay_buffer.next_note[:N]).to(device)
    note_bg_only       = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    next_note_bg_only  = torch.FloatTensor(replay_buffer.next_note_bg_only[:N]).to(device)
    state      = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    next_state = torch.FloatTensor(replay_buffer.next_state[:N]).to(device)
    action     = torch.LongTensor(replay_buffer.action[:N]).to(device)
    reward     = torch.FloatTensor(replay_buffer.reward[:N]).to(device)
    bc_prob    = torch.FloatTensor(replay_buffer.bc_prob[:N]).to(device)

    dataset = CustomDatasetForDR(
        note, next_note, state, next_state, action, reward,
        torch.FloatTensor(replay_buffer.done[:N]), bc_prob, note_bg_only, next_note_bg_only
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn_for_DR
    )

    dr_vals = []
    eps = 1e-8
    pos = 0 

    with torch.no_grad():
        for note_b, next_note_b, state_b, next_state_b, action_b, reward_b, done_b, bc_prob_b, note_bg_only, next_note_bg_only in loader:
            B = note_b.size(0)
            note_b       = note_b.to(device)
            next_note_b  = next_note_b.to(device)
            state_b      = state_b.to(device)
            next_state_b = next_state_b.to(device)
            action_b     = action_b.to(device).long()
            reward_b     = reward_b.to(device).squeeze(-1)
            done_b       = done_b.to(device).squeeze(-1)
            b_prob       = bc_prob_b.to(device).squeeze(-1) + eps

            note_bg_only       = note_bg_only.to(device)
            next_note_bg_only  = next_note_bg_only.to(device)

            # Q, π, V 
            # q, _, _    = policy.Q(state_b, note_b)
            q = pred_q_value(algorithm, policy, state_b, note_b, note_bg_only)
            pi         = F.softmax(q, dim=1)
            v_s        = (pi * q).sum(dim=1)
            q_sa       = q.gather(1, action_b).squeeze(1)

            # q_next, _, _ = policy.Q(next_state_b, next_note_b)
            q_next = pred_q_value(algorithm, policy, next_state_b, next_note_b, next_note_bg_only)
            pi_next      = F.softmax(q_next, dim=1)
            v_next_raw   = (pi_next * q_next).sum(dim=1)
            v_next       = v_next_raw * (1 - done_b)

            pi_a      = pi.gather(1, action_b).squeeze(1)
            rho_all   = torch.clamp(pi_a / b_prob, max=clip)
            delta     = reward_b + gamma * v_next - q_sa

            end_idxs = torch.nonzero(done_b, as_tuple=True)[0]
            if end_idxs.numel() > 0:
                start_idxs = torch.cat([
                    torch.tensor([0], device=device),
                    end_idxs[:-1] + 1
                ])
                for start, end in zip(start_idxs.tolist(), end_idxs.tolist()):
                    global_start = pos + start
                    if global_start != 0 and done_all_np[global_start - 1] == 0:
                        continue

                    rho_slice   = rho_all[start:end+1]
                    delta_slice = delta[start:end+1]
                    rho_cum     = torch.cumprod(rho_slice, dim=0)

                    dr_val = v_s[start].item()
                    for k in range(rho_cum.size(0)):
                        dr_val += (gamma ** k) * rho_cum[k].item() * delta_slice[k].item()
                    dr_vals.append(dr_val)

            pos += B


    dr_mean = float(np.mean(dr_vals))
    boot_means = []
    M = len(dr_vals)
    for _ in range(n_bootstrap):
        sample = np.random.choice(dr_vals, size=M, replace=True)
        boot_means.append(np.mean(sample))
    # np.save("./plot/radiology_bootstrap_dr.npy", boot_means)
    lower = np.percentile(boot_means, 100 * (alpha/2))
    upper = np.percentile(boot_means, 100 * (1 - alpha/2))

    print(f"Multi-Step DR estimate: {dr_mean:.4f}")
    print(f"{100*(1-alpha):.1f}% CI: [{lower:.4f}, {upper:.4f}]")

    return dr_mean, lower, upper


def eval_wis_ci(algorithm,
                policy,
                replay_buffer,
                clip: float = 5.0,
                gamma: float = 0.98,
                device: str = "cuda",
                n_bootstrap: int = 1000,
                alpha: float = 0.05):
 
    if algorithm == 'IQL':
        policy.critic1.to(device)
        policy.critic2.to(device)
        policy.critic1.eval()
        policy.critic2.eval()
    else:
        policy.Q.to(device)
        policy.Q.eval()
    eps = 1e-8

    N = replay_buffer.crt_size
    note    = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    note_bg_only       = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    state   = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    action  = torch.LongTensor(replay_buffer.action[:N]).to(device).squeeze(-1)
    reward  = torch.FloatTensor(replay_buffer.reward[:N]).to(device).squeeze(-1)
    done    = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)
    bc_prob = torch.FloatTensor(replay_buffer.bc_prob[:N]).to(device).squeeze(-1) + eps

    with torch.no_grad():
        # q     = policy.Q(state, note)[0]                # [N, A]
        q = pred_q_value(algorithm, policy, state, note, note_bg_only)

        pi    = F.softmax(q, dim=1)                     # [N, A]
        pi_a  = pi.gather(1, action.unsqueeze(1)).squeeze(1)  # [N]
        rho_t = torch.clamp(pi_a / bc_prob, max=clip)         # [N]

    end_idxs   = torch.nonzero(done, as_tuple=True)[0]
    start_idxs = torch.cat([
        torch.tensor([0], device=device),
        end_idxs[:-1] + 1
    ])

    weights = []
    returns = []

    for start, end in zip(start_idxs.tolist(), end_idxs.tolist()):
        rho_i = float(torch.prod(rho_t[start:end+1]).cpu().item())

        G_i = 0.0
        for k in range(end - start + 1):
            G_i += (gamma ** k) * float(reward[start + k].cpu().item())

        weights.append(rho_i)
        returns.append(G_i)

    weights = np.array(weights, dtype=np.float64)
    returns = np.array(returns, dtype=np.float64)

    if weights.sum() > 0:
        wis_mean = float((weights * returns).sum() / weights.sum())
    else:
        wis_mean = 0.0

    M = len(weights)
    boot_estimates = []
    for _ in range(n_bootstrap):
        idxs = np.random.randint(0, M, size=M)
        w = weights[idxs]
        r = returns[idxs]
        if w.sum() > 0:
            boot_estimates.append((w * r).sum() / w.sum())
        else:
            boot_estimates.append(0.0)

    # np.save("./plot/radiology_bootstrap_wis.npy", boot_estimates)
    lower = np.percentile(boot_estimates, 100 * (alpha / 2))
    upper = np.percentile(boot_estimates, 100 * (1 - alpha / 2))

    print(f"WIS estimate: {wis_mean:.4f}")
    print(f"{100*(1-alpha):.1f}% CI: [{lower:.4f}, {upper:.4f}]")

    return wis_mean, lower, upper



def eval_opera_ci(algorithm,
                  policy,
                  replay_buffer,
                  gamma: float = 0.98,
                  device: str = "cuda",
                  eta: float = 0.8,
                  n_weight_bootstrap: int = 500,
                  n_bootstrap: int = 1000,
                  alpha: float = 0.05):

    # 1) per-episode values
    _, _, fqe_vals = _compute_fqe_per_episode(algorithm, policy, replay_buffer,
                                              gamma, device)
    _, _, dr_vals  = _compute_dr_per_episode(algorithm, policy, replay_buffer,
                                             gamma, device)

    # align lengths
    M_common = min(len(fqe_vals), len(dr_vals))
    if M_common < len(fqe_vals) or M_common < len(dr_vals):
        print(f"[Warning] Truncating to common episode count: {M_common}")
    fqe_vals = np.array(fqe_vals[:M_common], dtype=np.float64)
    dr_vals  = np.array(dr_vals[:M_common],  dtype=np.float64)
    M = M_common

    # 2) mean vector (dimension 2)
    s_vec = np.array([
        fqe_vals.mean(),
        dr_vals.mean(),
    ])

    # 3) bootstrap for covariance estimation
    deltas = []
    n_eta = max(1, int(M * eta))
    for _ in range(n_weight_bootstrap):
        idxs = np.random.choice(M, size=n_eta, replace=True)
        s_b = np.array([
            fqe_vals[idxs].mean(),
            dr_vals[idxs].mean(),
        ])
        deltas.append(s_b - s_vec)
    deltas = np.stack(deltas, axis=0)   # shape (B,2)
    A = np.cov(deltas, rowvar=False, bias=True)  # (2×2)

    # 4) solve QP: minimize αᵀ A α subject to sum(α)=1
    k = 2
    alpha_var = cp.Variable(k)
    objective = cp.Minimize(cp.quad_form(alpha_var, A))
    constraints = [cp.sum(alpha_var) == 1]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    alphas = alpha_var.value

    opera_est = float(alphas.dot(s_vec))

    # 5) fixed-α bootstrap for CI
    opera_boot = []
    for _ in range(n_bootstrap):
        idxs = np.random.choice(M, size=M, replace=True)
        s_b = np.array([
            fqe_vals[idxs].mean(),
            dr_vals[idxs].mean(),
        ])
        opera_boot.append(alphas.dot(s_b))
    # np.save("./plot/radiology_bootstrap_opera.npy", opera_boot)
    lower = float(np.percentile(opera_boot, 100 * (alpha/2)))
    upper = float(np.percentile(opera_boot, 100 * (1 - alpha/2)))

    print(f"OPERA estimate (FQE+DR): {opera_est:.4f}")
    print(f"{100*(1-alpha):.1f}% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"Alphas (FQE, DR): {alphas}")

    return opera_est, lower, upper, alphas


# --------------------------------------------------
def _compute_fqe_per_episode(algorithm, policy, replay_buffer, gamma, device):
    """
    From eval_fqe_ci: return v0_list only.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    fqe_Q        = copy.deepcopy(policy)
    fqe_target_Q = copy.deepcopy(policy)
    if algorithm == 'IQL':
        fqe_Q.critic1.train();    fqe_target_Q.critic1.eval()
        q_params = fqe_Q.critic1.parameters()
    else:
        fqe_Q.Q.train();          fqe_target_Q.Q.eval()
        q_params = fqe_Q.Q.parameters()
    optimizer = torch.optim.Adam(q_params, lr=1e-3)
    criterion = torch.nn.MSELoss()

    # load data
    N = replay_buffer.crt_size
    note      = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    next_note = torch.FloatTensor(replay_buffer.next_note[:N]).to(device)
    bg_note   = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    bg_next   = torch.FloatTensor(replay_buffer.next_note_bg_only[:N]).to(device)
    state     = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    next_state= torch.FloatTensor(replay_buffer.next_state[:N]).to(device)
    action    = torch.LongTensor(replay_buffer.action[:N]).to(device)
    reward    = torch.FloatTensor(replay_buffer.reward[:N]).to(device)
    done      = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)

    dataset = FQEDataset(note, next_note, state, next_state,
                         action, reward, done, bg_note, bg_next)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)

    # train FQE
    for _ in range(20):
        for n, nn, s, ns, a, r, d, nbg, nnbg in loader:
            q_pred = pred_q_value(algorithm, fqe_Q, s, n, nbg)
            q_sa   = q_pred.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q_next = pred_q_value(algorithm, fqe_target_Q, ns, nn, nnbg)
                pi_next= F.softmax(q_next, dim=1)
                v_next = (pi_next * q_next).sum(dim=1) * (1-d)
                target = r + gamma * v_next
            loss = criterion(q_sa, target)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # hard update
        for p_o, p_t in zip(q_params,
                            fqe_target_Q.critic1.parameters() if algorithm=='IQL' else fqe_target_Q.Q.parameters()):
            p_t.data.copy_(p_o.data)

    # collect V0 per episode
    if algorithm == 'IQL':
        fqe_Q.critic1.eval()
    else:
        fqe_Q.Q.eval()

    v0_list = []
    with torch.no_grad():
        end_idxs   = torch.nonzero(done, as_tuple=True)[0]
        start_idxs = torch.cat([torch.tensor([0], device=device),
                                end_idxs[:-1]+1])
        for idx in start_idxs.tolist():
            s0  = state[idx].unsqueeze(0)
            n0  = note[idx].unsqueeze(0)
            bg0 = note[idx].unsqueeze(0)
            q0  = pred_q_value(algorithm, fqe_Q, s0, n0, bg0)
            pi0 = F.softmax(q0, dim=1)
            v0_list.append((pi0 * q0).sum(dim=1).item())

    return float(np.mean(v0_list)), (None, None), v0_list


def _compute_dr_per_episode(algorithm, policy, replay_buffer, gamma, device):
    """
    From eval_multi_step_doubly_robust_ci: return dr_vals only.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    if algorithm == 'IQL':
        policy.actor.to(device); policy.actor.eval()
    else:
        policy.Q.to(device); policy.Q.eval()

    N = replay_buffer.crt_size
    note      = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    next_note = torch.FloatTensor(replay_buffer.next_note[:N]).to(device)
    bg_note   = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    bg_next   = torch.FloatTensor(replay_buffer.next_note_bg_only[:N]).to(device)
    state     = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    next_state= torch.FloatTensor(replay_buffer.next_state[:N]).to(device)
    action    = torch.LongTensor(replay_buffer.action[:N]).to(device)
    reward    = torch.FloatTensor(replay_buffer.reward[:N]).to(device).squeeze(-1)
    done      = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)
    bc_prob   = torch.FloatTensor(replay_buffer.bc_prob[:N]).to(device).squeeze(-1) + 1e-8

    dataset = CustomDatasetForDR(note, next_note, state, next_state,
                                action, reward, torch.FloatTensor(replay_buffer.done[:N]),
                                bc_prob, bg_note, bg_next)
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        collate_fn=custom_collate_fn_for_DR)

    dr_vals = []
    pos = 0
    with torch.no_grad():
        for n_b, nn_b, s_b, ns_b, a_b, r_b, d_b, bc_b, nbg_b, nnbg_b in loader:
            n_b    = n_b.to(device)
            nn_b   = nn_b.to(device)
            s_b    = s_b.to(device)
            ns_b   = ns_b.to(device)
            a_b    = a_b.to(device)
            r_b    = r_b.to(device).squeeze(-1)
            d_b    = d_b.to(device).squeeze(-1)
            bc_b   = bc_b.to(device).squeeze(-1)
            nbg_b  = nbg_b.to(device)
            nnbg_b = nnbg_b.to(device)

            B = s_b.size(0)
            q      = pred_q_value(algorithm, policy, s_b, n_b, nnbg_b)
            pi     = F.softmax(q, dim=1)
            v_s    = (pi * q).sum(dim=1)
            q_sa   = q.gather(1, a_b.long()).squeeze(1)

            q_next = pred_q_value(algorithm, policy, ns_b, nn_b, nnbg_b)
            pi_n   = F.softmax(q_next, dim=1)
            v_next = (pi_n * q_next).sum(dim=1) * (1 - d_b)

            rho_all= torch.clamp(pi.gather(1, a_b).squeeze(1)/bc_b, max=5.0)
            delta  = r_b + gamma*v_next - q_sa

            end_idxs = torch.nonzero(d_b, as_tuple=True)[0]
            if end_idxs.numel()>0:
                starts = torch.cat([torch.tensor([0], device=device),
                                    end_idxs[:-1]+1])
                for st, en in zip(starts.tolist(), end_idxs.tolist()):
                    global_start = pos + st
                    if global_start!=0 and replay_buffer.done[global_start-1]==0:
                        continue

                    rho_slice   = rho_all[st:en+1]
                    delta_slice = delta[st:en+1]
                    rho_cum     = torch.cumprod(rho_slice, dim=0)

                    # flatten delta to list
                    delta_list = delta_slice.view(-1).cpu().tolist()

                    dr_val = float(v_s[st].item())
                    for k, d_k in enumerate(delta_list):
                        dr_val += (gamma**k) * float(rho_cum[k].item()) * float(d_k)
                    dr_vals.append(dr_val)
            pos += B

    return float(np.mean(dr_vals)), (None, None), dr_vals


def _compute_wis_per_episode(algorithm, policy, replay_buffer, gamma, device):
    """
    From eval_wis_ci: return weighted_returns = [ρ_i * G_i] per episode.
    """
    import numpy as np

    if algorithm == 'IQL':
        policy.critic1.to(device); policy.critic2.to(device)
        policy.critic1.eval();   policy.critic2.eval()
    else:
        policy.Q.to(device);     policy.Q.eval()

    N = replay_buffer.crt_size
    note    = torch.FloatTensor(replay_buffer.note[:N]).to(device)
    bg_note = torch.FloatTensor(replay_buffer.note_bg_only[:N]).to(device)
    state   = torch.FloatTensor(replay_buffer.state[:N]).to(device)
    action  = torch.LongTensor(replay_buffer.action[:N]).to(device).squeeze(-1)
    reward  = torch.FloatTensor(replay_buffer.reward[:N]).to(device).squeeze(-1)
    done    = torch.FloatTensor(replay_buffer.done[:N]).to(device).squeeze(-1)
    bc_prob = torch.FloatTensor(replay_buffer.bc_prob[:N]).to(device).squeeze(-1) + 1e-8

    with torch.no_grad():
        q    = pred_q_value(algorithm, policy, state, note, bg_note)
        pi   = F.softmax(q, dim=1)
        pi_a = pi.gather(1, action.unsqueeze(1)).squeeze(1)
        rho  = torch.clamp(pi_a / bc_prob, max=5.0)

    end_idxs   = torch.nonzero(done, as_tuple=True)[0]
    start_idxs = torch.cat([torch.tensor([0], device=device),
                            end_idxs[:-1]+1])

    weighted_returns = []
    for st, en in zip(start_idxs.tolist(), end_idxs.tolist()):
        rho_i = float(torch.prod(rho[st:en+1]).cpu().item())
        G_i   = sum((gamma**k) * float(reward[st+k].cpu().item())
                    for k in range(en-st+1))
        weighted_returns.append(rho_i * G_i)

    wis_mean = float(np.mean(weighted_returns))
    return wis_mean, weighted_returns, (None, None)






def _as_1d_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)

def _actions_to_vaso_iv(actions_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _as_1d_np(actions_1d).astype(np.int64)
    vaso = (a % 5) + 1
    iv = a // 5
    return vaso, iv


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))

    y01 = (y > 0.5).astype(int)
    mean1 = float(np.mean(x[y01 == 1])) if np.any(y01 == 1) else np.nan
    mean0 = float(np.mean(x[y01 == 0])) if np.any(y01 == 0) else np.nan
    std_all = float(np.std(x, ddof=1)) if n >= 2 else np.nan

    r, p = np.nan, np.nan
    if n >= 3 and (np.unique(x).size > 1) and (np.unique(y).size > 1):
        try:
            from scipy.stats import spearmanr
            rr = spearmanr(x, y)
            r = float(rr.correlation) if hasattr(rr, "correlation") else float(rr[0])
            p = float(rr.pvalue)       if hasattr(rr, "pvalue")      else float(rr[1])
        except Exception:
            import pandas as pd
            xr = pd.Series(x).rank(method="average").to_numpy()
            yr = pd.Series(y).rank(method="average").to_numpy()
            xr = xr - xr.mean()
            yr = yr - yr.mean()
            denom = np.sqrt(np.sum(xr * xr) * np.sum(yr * yr))
            r = float(np.sum(xr * yr) / denom) if denom > 0 else np.nan
            p = np.nan

    return {
        "r": r,
        "p": p,
        "n": n,
        "mean1": mean1,
        "mean0": mean0,
        "std": std_all,
        "method": "spearman",
    }


def eval_action_discrepancy_spearman(
    algorithm,
    policy,
    replay_buffer,
    device: str = "cuda",
    batch_size: int = 8181,
    done_threshold: float = 0.5,
    require_terminal_reward_pm1: bool = True,
    verbose: bool = True,
):

    N = int(replay_buffer.crt_size)

    notes = torch.tensor(replay_buffer.note[:N], dtype=torch.float32, device=device)
    notes_bg_only = torch.tensor(replay_buffer.note_bg_only[:N], dtype=torch.float32, device=device)
    states = torch.tensor(replay_buffer.state[:N], dtype=torch.float32, device=device)

    acts_buf = _as_1d_np(replay_buffer.action[:N]).astype(np.int64)
    rewards = _as_1d_np(replay_buffer.reward[:N]).astype(float)
    dones = _as_1d_np(replay_buffer.done[:N]).astype(float)

    if algorithm == "IQL":
        policy.actor.eval()
    else:
        policy.Q.eval()

    ai_actions = []
    with torch.no_grad():
        for st in range(0, N, batch_size):
            ed = min(N, st + batch_size)
            pa = pred_action_batch_safe(algorithm, policy, states[st:ed], notes[st:ed], notes_bg_only[st:ed])

            if torch.is_tensor(pa):
                pa_t = pa
                if pa_t.dim() > 1:
                    pa_t = pa_t.squeeze(-1)
                pa_np = pa_t.detach().long().cpu().numpy().reshape(-1)
            else:
                pa_np = np.asarray(pa).reshape(-1)

            ai_actions.append(pa_np)

    acts_pi = np.concatenate(ai_actions, axis=0).astype(np.int64)
    if len(acts_pi) != N:
        raise RuntimeError(f"Predicted action length mismatch: got {len(acts_pi)} vs N={N}")

    x_vaso, x_iv, x_comb, y = [], [], [], []
    start = 0
    n_incomplete_tail = 0

    for t in range(N):
        if dones[t] > done_threshold:
            end = t
            r = float(rewards[end])

            if require_terminal_reward_pm1 and (r not in (-1.0, 1.0)):
                start = end + 1
                continue

            y_i = (r + 1.0) / 2.0  # {0,1}
            buf_ep = acts_buf[start:end + 1]
            pi_ep = acts_pi[start:end + 1]

            vaso_buf, iv_buf = _actions_to_vaso_iv(buf_ep)
            vaso_pi, iv_pi = _actions_to_vaso_iv(pi_ep)

            vaso_diff = np.abs(vaso_buf - vaso_pi).astype(float)
            iv_diff   = np.abs(iv_buf   - iv_pi).astype(float)

            x_vaso.append(float(np.mean(vaso_diff)) if vaso_diff.size else np.nan)
            x_iv.append(float(np.mean(iv_diff)) if iv_diff.size else np.nan)
            x_comb.append(float(np.mean(vaso_diff + iv_diff)) if (vaso_diff.size and iv_diff.size) else np.nan)

            y.append(float(y_i))
            start = end + 1

    if start < N:
        n_incomplete_tail = 1

    x_vaso = np.asarray(x_vaso, dtype=float)
    x_iv   = np.asarray(x_iv, dtype=float)
    x_comb = np.asarray(x_comb, dtype=float)
    y      = np.asarray(y, dtype=float)

    # 3) Spearman correlation
    res_vaso = spearman_corr(x_vaso, y)
    res_iv   = spearman_corr(x_iv, y)
    res_comb = spearman_corr(x_comb, y)

    out = {
        "vaso": res_vaso,
        "iv": res_iv,
        "comb": res_comb,
        "n_episodes": int(len(y)),
        "n_incomplete_tail": int(n_incomplete_tail),
        "x_vaso": x_vaso,
        "x_iv": x_iv,
        "x_comb": x_comb,
        "y": y,
    }

    if verbose:
        print(f"[Spearman|mean] vaso: r={out['vaso']['r']:.4f}, p={out['vaso']['p']}, n={out['vaso']['n']}")
        print(f"  - mean(survival)= {out['vaso']['mean1']:.4f}, mean(death)= {out['vaso']['mean0']:.4f}, std(all)= {out['vaso']['std']:.4f}")

        print(f"[Spearman|mean] iv  : r={out['iv']['r']:.4f}, p={out['iv']['p']}, n={out['iv']['n']}")
        print(f"  - mean(survival)= {out['iv']['mean1']:.4f}, mean(death)= {out['iv']['mean0']:.4f}, std(all)= {out['iv']['std']:.4f}")

        print(f"[Spearman|mean] comb: r={out['comb']['r']:.4f}, p={out['comb']['p']}, n={out['comb']['n']}")
        print(f"  - mean(survival)= {out['comb']['mean1']:.4f}, mean(death)= {out['comb']['mean0']:.4f}, std(all)= {out['comb']['std']:.4f}")

        if n_incomplete_tail:
            print("[warn]")

    return out


def pred_action_batch_safe(algorithm, policy, state, note, note_bg_only):

    with torch.no_grad():
        if algorithm in ["CQL", "CQL_Cross_Attention", "CQL_Cross_Context_Attention", "CQL_Text"]:
            if algorithm == "CQL":
                q = policy.Q(state)
            elif algorithm == "CQL_Cross_Attention":
                q = policy.Q(state, note)
            elif algorithm == "CQL_Cross_Context_Attention":
                q = policy.Q(state, note, note_bg_only)
            else:  # "CQL_Text"
                q = policy.Q(note)
            return q.argmax(1).long()

        if algorithm == "IQL":
            try:
                q = pred_q_value(algorithm, policy, state, note, note_bg_only)
                return q.argmax(1).long()
            except Exception:
                acts = []
                B = state.shape[0]
                for i in range(B):
                    ai = pred_action(algorithm, policy,
                                    state[i:i+1],
                                    note[i:i+1] if note is not None else note,
                                    note_bg_only[i:i+1] if note_bg_only is not None else note_bg_only)
                    if torch.is_tensor(ai):
                        ai = int(ai.reshape(-1)[0].item())
                    else:
                        ai = int(np.asarray(ai).reshape(-1)[0])
                    acts.append(ai)
                return torch.tensor(acts, device=state.device, dtype=torch.long)

        acts = []
        B = state.shape[0]
        for i in range(B):
            ai = pred_action(algorithm, policy,
                             state[i:i+1],
                             note[i:i+1] if note is not None else note,
                             note_bg_only[i:i+1] if note_bg_only is not None else note_bg_only)
            if torch.is_tensor(ai):
                ai = int(ai.reshape(-1)[0].item())
            else:
                ai = int(np.asarray(ai).reshape(-1)[0])
            acts.append(ai)
        return torch.tensor(acts, device=state.device, dtype=torch.long)