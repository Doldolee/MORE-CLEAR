from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch
from tqdm import tqdm
from collections import deque
from tqdm import tqdm
from typing import List, Union

## only extracts background
def extract_background_next_notes(next_notes, notes, done, missing='no clinical note'):

    backgrounds = []
    first_valid = None

    for nxt, cur, is_done in zip(next_notes, notes, done):
        if first_valid is None and nxt != missing:
            first_valid = nxt

        if first_valid is None:
            backgrounds.append(missing)
        else:
            backgrounds.append(f"[context] {first_valid}")
        if is_done:
            first_valid = None

    return backgrounds
    
# only extracts background
def extract_background_states(notes, done, missing='no clinical note'):
    """
    notes: list of str
    done: list of bool/int, True at episode boundaries
    missing: placeholder for missing notes
    returns: list of str, each entry is either "[context] first_valid_note" or the missing placeholder
    """
    backgrounds = []
    first_valid = None

    for note, is_done in zip(notes, done):
        if first_valid is None and note != missing:
            first_valid = note


        if first_valid is None:
            backgrounds.append(missing)
        else:
            backgrounds.append(f"[context] {first_valid}")


        if is_done:
            first_valid = None

    return backgrounds

## background + window 3 stack
def bg_stack_next_note(
    next_notes: List[str],
    done: List[Union[bool, int]],
    sep: str = ' | ',
    missing: str = 'no clinical note'
) -> List[str]:
    """
    Episode-aware stacking for next_notes, including only t-1 and t-2 within the same episode.
    """
    stacked = []
    first_valid = None
    last_reset = -1

    for i, (nxt, flag) in enumerate(zip(next_notes, done)):
        flag = bool(flag)
        is_first = False

        # Detect first valid note in episode
        if first_valid is None and nxt != missing:
            first_valid = nxt
            is_first = True

        # Collect t-2, t-1 valid notes within current episode
        suffix = []
        for idx in (i-2, i-1):
            if idx > last_reset and 0 <= idx < len(next_notes):
                prev = next_notes[idx]
                if prev != missing and prev != first_valid and prev not in suffix:
                    suffix.append(prev)

        # Build output
        if first_valid is None:
            out = nxt
        else:
            if is_first:
                out = f"[current status] {nxt}"
            else:
                prefix = f"[background] {first_valid}"
                if suffix:
                    prefix += sep + sep.join(suffix)
                if nxt != missing:
                    out = f"{prefix} || [current status] {nxt}"
                else:
                    out = prefix

        stacked.append(out)

        # Reset at episode boundary
        if flag:
            first_valid = None
            last_reset = i

    return stacked

def bg_stack_note(notes: List[str],
                          done: List[Union[bool, int]],
                          sep: str = ' | ',
                          missing: str = 'no clinical note'
                         ) -> List[str]:
    stacked = []
    first_valid = None

    for i, (note, flag) in enumerate(zip(notes, done)):
        flag = bool(flag)
        
        if first_valid is None:
            if note != missing:
                first_valid = note
                out = f"[current status] {note}"
            else:
                out = note
            stacked.append(out)
            if flag:
                first_valid = None
            continue
        
    
        prev_idxs = [i-2, i-1]
        suffix = []
        for idx in prev_idxs:
            if 0 <= idx < len(notes):
                prev_note = notes[idx]
                if prev_note != missing and prev_note != first_valid:
                    if prev_note not in suffix:
                        suffix.append(prev_note)
        
        prefix = f"[background] {first_valid}"
        
        if note != missing:
       
            prefix_ext = prefix + (sep + sep.join(suffix) if suffix else "")
            out = f"{prefix_ext} || [current status] {note}"
        else:
      
            out = prefix + (f" || {sep.join(suffix)}" if suffix else "")
        
        stacked.append(out)
        if flag:
            first_valid = None
    
    return stacked
    
## background
def impute_next_notes_with_background(next_notes, notes, done, missing='no clinical note'):

    imputed = []
    first_valid = None

    for nxt, cur, is_done in zip(next_notes, notes, done):
        is_first = False
        if first_valid is None and nxt != missing:
            first_valid = f"[background] {nxt}"
            is_first = True

        if first_valid is not None:
            if is_first:
                imputed_note = first_valid
            else:
                if nxt == missing:
                    imputed_note = first_valid
                else:
                    imputed_note = f"{first_valid} || {nxt}"
        else:
            imputed_note = nxt

        imputed.append(imputed_note)

        if is_done:
            first_valid = None

    return imputed

## background
def impute_notes_with_background(notes, done, missing='no clinical note'):

    imputed = []
    first_valid = None

    for note, is_done in zip(notes, done):
        if first_valid is None and note != missing:
            first_valid = f"[background] {note}"

        if first_valid is not None:
            if note == missing:
                imputed_note = first_valid
            else:
                imputed_note = f"{first_valid} || {note}"
        else:
            imputed_note = note

        imputed.append(imputed_note)

        if is_done:
            first_valid = None

    return imputed
    
## simple impute
def impute_notes(notes, done, missing='no clinical note'):
    """
    notes: list of str
    done: list of bool/int, True at episode boundaries
    missing: 결측을 나타내는 문자열
    returns: imputed list of str
    """
    imputed = []
    last_valid = None

    for note, is_done in zip(notes, done):
        if note != missing:
            imputed.append(note)
            last_valid = note
        else:
            imputed.append(last_valid if last_valid is not None else note)

        if is_done:
            last_valid = None

    return imputed

## simple impute
def impute_next_notes(next_notes, notes, done, missing='no clinical note'):
    """
    next_notes: list of str, aligned with `done`, where next_notes[i] is the note at the next time step
    notes:      list of str, aligned one step behind next_notes (so notes[i] is the “current” note for next_notes[i])
    done:       list of bool/int, True (or 1) at the end of each episode, aligned with next_notes
    missing:    the placeholder string indicating a missing clinical note

    Returns a new list where each missing next_note is imputed by:
      1) the current note at the same index, if available
      2) else the most recent valid imputed note within the same episode
      3) else left as missing (if no valid note seen yet)
    """
    imputed = []
    last_valid = None

    for idx, (nxt, cur, is_done) in enumerate(zip(next_notes, notes, done)):
        if nxt != missing:
            imputed.append(nxt)
            last_valid = nxt
        else:
            if cur != missing:
                imputed.append(cur)
                last_valid = cur
            else:
                imputed.append(last_valid if last_valid is not None else missing)

        if is_done:
            last_valid = None

    return imputed

def stack_notes(notes, done, window=3, sep=' | ', missing='no clinical note'):

    stacked = []
    dq = deque(maxlen=window)

    for note, is_done in tqdm(zip(notes, done)):
        dq.append(note)

        valid = [n for n in dq if n != missing]
        stacked.append(sep.join(valid) if valid else '')

        if is_done:
            dq.clear()

    return stacked

    
def stack_next_notes(next_notes, notes, done, window=3, sep=' | ', missing='no clinical note'):

    stacked = []
    dq = deque(maxlen=window)

    for nxt, cur, is_done in tqdm(zip(next_notes, notes, done)):
        if is_done:
            dq.clear()

        dq.append(nxt)

        valid = [n for n in dq if n != missing]
        stacked.append(sep.join(valid) if valid else '')

    return stacked

def custom_collate_fn(batch):
    notes, dems, states, actions, lengths, times, rewards = zip(*batch)
    batch_dem = torch.stack(dems)
    batch_states = torch.stack(states)
    batch_actions = torch.stack(actions)
    batch_lengths = torch.stack(lengths)
    batch_times = torch.stack(times)
    batch_rewards = torch.stack(rewards)
    return np.array(notes, dtype=object), batch_dem, batch_states, batch_actions, batch_lengths, batch_times, batch_rewards

class CustomDataset(Dataset):
    def __init__(self, train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards):

        self.train_note = train_note  

        self.train_dem = train_dem.clone().detach() if isinstance(train_dem, torch.Tensor) else torch.tensor(train_dem, dtype=torch.float32)
        self.train_states = train_states.clone().detach() if isinstance(train_states, torch.Tensor) else torch.tensor(train_states, dtype=torch.float32)
        self.train_actions = train_actions.clone().detach() if isinstance(train_actions, torch.Tensor) else torch.tensor(train_actions, dtype=torch.float32)
        self.train_lengths = train_lengths.clone().detach() if isinstance(train_lengths, torch.Tensor) else torch.tensor(train_lengths, dtype=torch.float32)
        self.train_times = train_times.clone().detach() if isinstance(train_times, torch.Tensor) else torch.tensor(train_times, dtype=torch.float32)
        self.train_rewards = train_rewards.clone().detach() if isinstance(train_rewards, torch.Tensor) else torch.tensor(train_rewards, dtype=torch.float32)

    def __len__(self):
        return len(self.train_note)

    def __getitem__(self, index):
        note = self.train_note[index]
        dem = self.train_dem[index]
        state = self.train_states[index]
        action = self.train_actions[index]
        length = self.train_lengths[index]
        time = self.train_times[index]
        reward = self.train_rewards[index]
        return note, dem, state, action, length, time, reward
        
class ReplayBuffer(object):
    def __init__(self, 
                 state_dim,
                 batch_size,
                 buffer_size,
                 device,
                 data_path,
                 buffer_path
                 ):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.data_path = data_path
        self.buffer_path = buffer_path
        
    
        self.ptr = 0
        self.crt_size = 0

        self.note = np.empty((self.max_size, 1), dtype=object)
        self.next_note = np.empty((self.max_size, 1), dtype=object)

        self.state = np.zeros((self.max_size, state_dim))
        self.next_state = np.array(self.state)
        self.action = np.zeros((self.max_size, 1))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))

    def add(self, notes, next_notes, states, action, next_state, reward, done):
        self.note[self.ptr] = notes
        self.next_note[self.ptr] = next_notes
        self.state[self.ptr] = states
        self.next_state[self.ptr] = next_state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)
        
    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)

        return(
            self.note[ind],
            self.next_note[ind],
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )
     
    def save(self, only_test_set = False):
        if only_test_set:
            flag = "test"
        else:
            flag = "train_val"
        np.save(f"{self.buffer_path}{flag}_note.npy", self.note[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_next_note.npy", self.next_note[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_state.npy", self.state[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_action.npy", self.action[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{self.buffer_path}{flag}_done.npy", self.done[:self.crt_size])


    def load(self, size=-1, only_test_set = False):
        if only_test_set:
            flag = "test"
        else:
            flag = "train_val"

        reward_buffer = np.load(f"{self.buffer_path}{flag}_reward.npy")
      
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_note.npy", allow_pickle=True)[:self.crt_size]
        self.next_note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_next_note.npy", allow_pickle=True)[:self.crt_size]
        self.state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.done[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_done.npy")[:self.crt_size]
        print(f"Replay Buffer loaded with {self.crt_size} elements.")

        
    
    def load_initial_data(self, only_test_set = False):
    
        train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards = torch.load(self.data_path["train"], weights_only = False)
        train_dataset = CustomDataset(train_note, train_dem, train_states, train_actions, train_lengths, train_times, train_rewards)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        val_note, val_dem, val_states, val_actions, val_lengths, val_times, val_rewards = torch.load(self.data_path["val"], weights_only = False)
        val_dataset = CustomDataset(val_note, val_dem, val_states, val_actions, val_lengths, val_times, val_rewards)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        test_note, test_dem, test_states, test_actions, test_lengths, test_times, test_rewards = torch.load(self.data_path["test"], weights_only = False)
        test_dataset = CustomDataset(test_note, test_dem, test_states, test_actions, test_lengths, test_times, test_rewards)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        if only_test_set:
            all_loaders_list = [test_loader]
        else:
            all_loaders_list = [train_loader, val_loader]

        for idx, loader in enumerate(all_loaders_list):
            for note, dem, state, action, length, time, reward in tqdm(loader):
                # print(note.shape, dem.shape, state.shape, action.shape, length.shape, time.shape, reward.shape)
                note = note
                dem = dem.to(self.device)
                state = state.to(self.device)
                action = action.to(self.device)
                length = length.to(self.device)
                time = time.to(self.device)
                reward = reward.to(self.device)
            
                max_length = int(length.max().item())
                note = note[:,:max_length,:]
                state = state[:,:max_length,:]
                dem = dem[:,:max_length,:]
                action = action[:,:max_length,:]
                reward = reward[:,:max_length]

                cur_notes, next_notes = note[:,:-1,:], note[:,1:,:]   
                cur_states, next_states = state[:,:-1,:], state[:,1:,:]                
                cur_dem, next_dem = dem[:,:-1,:], dem[:,1:,:]
                cur_actions = action[:,:-1,:]
                cur_rewards = reward[:,1:] 
      
                for batch in range(cur_states.shape[0]):
                    for i_trans in range(cur_states.shape[1]):
                        done = cur_rewards[batch,i_trans] != 0 
                        self.add(notes = cur_notes[batch,i_trans],
                                next_notes = next_notes[batch,i_trans],
                                states=torch.cat((cur_states[batch,i_trans],cur_dem[batch,i_trans]),dim=-1).cpu().numpy(), 
                                action=cur_actions[batch,i_trans].cpu().argmax().item(), # scalar
                                next_state=torch.cat((next_states[batch,i_trans], next_dem[batch,i_trans]), dim=-1).cpu().numpy(), 
                                reward=cur_rewards[batch,i_trans].cpu().item(), 
                                done=int(done.item()) 
                                )
                        if done:
                            break
        print("only test set? : ", only_test_set)
        print(self.ptr, self.crt_size)
        self.save(only_test_set = only_test_set)