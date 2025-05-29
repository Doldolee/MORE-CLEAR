import random
import numpy as np
import torch
from torch.utils.data import Dataset


class CustomDatasetForMortality(Dataset):
    def __init__(self, note, state, action, reward, done):

        self.note = note.clone().detach() if isinstance(note, torch.Tensor) else torch.tensor(note, dtype=torch.float32)
        self.state = state.clone().detach() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        self.action = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32)
        self.reward = reward.clone().detach() if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32)
        self.done = done.clone().detach() if isinstance(done, torch.Tensor) else torch.tensor(done, dtype=torch.float32)


    def __len__(self):
        return len(self.note)

    def __getitem__(self, index):
        note = self.note[index]
        state = self.state[index]
        action = self.action[index]
        reward = self.reward[index]
        done = self.done[index]

        return note, state, action, reward, done

def custom_collate_fn_for_mortality(batch):
    note, state, action, reward, done = zip(*batch)
    batch_notes = torch.stack(note)
    batch_states = torch.stack(state)
    batch_action = torch.stack(action)
    batch_reward = torch.stack(reward)
    batch_done = torch.stack(done)

    return batch_notes, batch_states, batch_action, batch_reward, batch_done


class FQEDataset(Dataset):
    def __init__(self, note, next_note, state, next_state, action, reward, done, note_bg_only, next_note_bg_only):
        self.note       = note
        self.next_note  = next_note
        self.note_bg_only       = note_bg_only
        self.next_note_bg_only  = next_note_bg_only
        self.state      = state
        self.next_state = next_state
        self.action     = action.squeeze(-1)
        self.reward     = reward.squeeze(-1)
        self.done       = done.squeeze(-1)

    def __len__(self):
        return self.state.size(0)

    def __getitem__(self, idx):
        return (self.note[idx],
                self.next_note[idx],
                self.state[idx],
                self.next_state[idx],
                self.action[idx],
                self.reward[idx],
                self.done[idx],
                self.note_bg_only[idx],
                self.next_note_bg_only[idx])

class CustomDatasetForDR(Dataset):
    def __init__(self, note, next_note, state, next_state, action, reward, done, bc_prob, note_bg_only, next_note_bg_only):
     
        self.note = note.clone().detach() if isinstance(note, torch.Tensor) else torch.tensor(note, dtype=torch.float32)
        self.next_note = next_note.clone().detach() if isinstance(next_note, torch.Tensor) else torch.tensor(next_note, dtype=torch.float32)
        self.state = state.clone().detach() if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32)
        self.next_state = next_state.clone().detach() if isinstance(next_state, torch.Tensor) else torch.tensor(next_state, dtype=torch.float32)
        self.action = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float32)
        self.reward = reward.clone().detach() if isinstance(reward, torch.Tensor) else torch.tensor(reward, dtype=torch.float32)
        self.done = done.clone().detach() if isinstance(done, torch.Tensor) else torch.tensor(done, dtype=torch.float32)
        self.bc_prob = bc_prob.clone().detach() if isinstance(bc_prob, torch.Tensor) else torch.tensor(bc_prob, dtype=torch.float32)
        self.note_bg_only = note_bg_only.clone().detach() if isinstance(note_bg_only, torch.Tensor) else torch.tensor(note_bg_only, dtype=torch.float32)
        self.next_note_bg_only = next_note_bg_only.clone().detach() if isinstance(next_note_bg_only, torch.Tensor) else torch.tensor(next_note_bg_only, dtype=torch.float32)


    def __len__(self):
        return len(self.note)

    def __getitem__(self, index):
        note = self.note[index]
        next_note = self.next_note[index]
        state = self.state[index]
        next_state = self.next_state[index]
        action = self.action[index]
        reward = self.reward[index]
        done = self.done[index]
        bc_prob = self.bc_prob[index]
        note_bg_only = self.note_bg_only[index]
        next_note_bg_only = self.next_note_bg_only[index]

        return note, next_note, state, next_state, action, reward, done, bc_prob, note_bg_only, next_note_bg_only

def custom_collate_fn_for_DR(batch):
    note, next_note, state, next_state, action, reward, done, bc_prob, note_bg_only, next_note_bg_only = zip(*batch)
    
    batch_note = torch.stack(note)
    batch_next_note = torch.stack(next_note)
    batch_states = torch.stack(state)
    batch_next_state = torch.stack(next_state)
    batch_action = torch.stack(action)
    batch_reward = torch.stack(reward)
    batch_done = torch.stack(done)
    batch_bc_prob = torch.stack(bc_prob)
    batch_note_bg_only = torch.stack(note_bg_only)
    batch_next_note_bg_only = torch.stack(next_note_bg_only)

    return batch_note, batch_next_note, batch_states, batch_next_state, batch_action, batch_reward, batch_done, batch_bc_prob, batch_note_bg_only, batch_next_note_bg_only


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True