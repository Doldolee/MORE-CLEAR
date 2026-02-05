import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self,
                 state_dim,
                 embed_dim,
                 batch_size,
                 target_data,
                 buffer_path,
                 note_form,
                 buffer_size=200000,
                 device='cuda'):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.target_data = target_data
        self.buffer_path = buffer_path
        self.embed_dim = embed_dim
        self.note_form = note_form

        self.ptr = 0
        self.crt_size = 0

        self.note = np.zeros((self.max_size, self.embed_dim))
        self.next_note = np.zeros((self.max_size, self.embed_dim))
        self.note_bg_only = np.zeros((self.max_size, self.embed_dim))
        self.next_note_bg_only = np.zeros((self.max_size, self.embed_dim))

        self.state = np.zeros((self.max_size, state_dim))
        self.next_state = np.array(self.state)
        self.action = np.zeros((self.max_size, 1))
        self.reward = np.zeros((self.max_size, 1))
        self.done = np.zeros((self.max_size, 1))
        self.bc_prob = np.zeros((self.max_size, 1))

    @staticmethod
    def _parse_frac(frac):
        """Accept 0.2 or '20%'."""
        if frac is None:
            return None
        if isinstance(frac, str):
            s = frac.strip()
            if s.endswith("%"):
                return float(s[:-1]) / 100.0
            return float(s)
        return float(frac)

    def load_original_dataset(
        self,
        size=-1,
        only_test_set=False,
        train_frac=None,   
        seed=0,
        shuffle=True
    ):

        flag = "test" if only_test_set else "train_val"
        reward_path = f"{self.buffer_path}{flag}_reward.npy"
        reward_mm = np.load(reward_path, mmap_mode="r")
        total_n = reward_mm.shape[0]

        frac = self._parse_frac(train_frac)
        if (not only_test_set) and (frac is not None):
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"train_frac must be in (0,1] or 'xx%'. Got: {train_frac}")
            target_n = max(1, int(total_n * frac))
        else:
            target_n = total_n

        if size is not None and size > 0:
            target_n = min(target_n, int(size))

        target_n = min(target_n, self.max_size)
        self.crt_size = int(target_n)

        if (not only_test_set) and (frac is not None):
            rng = np.random.default_rng(seed)
            if shuffle:
                idx = rng.choice(total_n, size=self.crt_size, replace=False)
                idx = np.sort(idx)  
            else:
                idx = np.arange(self.crt_size)
        else:
            idx = slice(0, self.crt_size)

        def load_arr(path, idx_):
            mm = np.load(path, mmap_mode="r")
            return np.array(mm[idx_])  

        self.reward[:self.crt_size] = np.array(reward_mm[idx])

        self.note[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}{self.note_form}_note_embedding.npy", idx
        )
        self.next_note[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}{self.note_form}_next_note_embedding.npy", idx
        )

        self.note_bg_only[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_impute_bg_only_note_embedding.npy", idx
        )
        self.next_note_bg_only[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_impute_bg_only_next_note_embedding.npy", idx
        )

        self.state[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_state.npy", idx
        )
        self.action[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_action.npy", idx
        )
        self.next_state[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_next_state.npy", idx
        )
        self.done[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_done.npy", idx
        )
        self.bc_prob[:self.crt_size] = load_arr(
            f"{self.buffer_path}{flag}_BC_prob.npy", idx
        )

        self.note = self.note[:self.crt_size].copy()
        self.next_note = self.next_note[:self.crt_size].copy()
        self.note_bg_only = self.note_bg_only[:self.crt_size].copy()
        self.next_note_bg_only = self.next_note_bg_only[:self.crt_size].copy()

        self.state = self.state[:self.crt_size].copy()
        self.next_state = self.next_state[:self.crt_size].copy()
        self.action = self.action[:self.crt_size].copy()
        self.reward = self.reward[:self.crt_size].copy()
        self.done = self.done[:self.crt_size].copy()
        self.bc_prob = self.bc_prob[:self.crt_size].copy()

        print(self.note.shape, self.next_note.shape, self.state.shape, self.action.shape,
              self.next_state.shape, self.reward.shape, self.done.shape)
        print(f"{self.target_data} Replay Buffer loaded with {self.crt_size} elements. (flag={flag}, frac={train_frac})")
        return self

    def load_validation_dataset(self, ori_data, size=-1):
        flag = f'scaled_{ori_data}_test'

        reward_buffer = np.load(f"{self.buffer_path}{flag}_reward.npy")
      
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)
        self.note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}{self.note_form}_note_embedding.npy")[:self.crt_size]
        self.next_note[:self.crt_size] = np.load(f"{self.buffer_path}{flag}{self.note_form}_next_note_embedding.npy")[:self.crt_size]

        self.note_bg_only[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_impute_bg_only_note_embedding.npy")[:self.crt_size]
        self.next_note_bg_only[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_impute_bg_only_next_note_embedding.npy")[:self.crt_size]

        self.state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.done[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_done.npy")[:self.crt_size]
        self.bc_prob[:self.crt_size] = np.load(f"{self.buffer_path}{flag}_BC_prob.npy")[:self.crt_size]

        self.note       = self.note[:self.crt_size].copy()
        self.next_note  = self.next_note[:self.crt_size].copy()
        self.state      = self.state[:self.crt_size].copy()
        self.next_state = self.next_state[:self.crt_size].copy()
        self.action     = self.action[:self.crt_size].copy()
        self.reward     = self.reward[:self.crt_size].copy()
        self.done       = self.done[:self.crt_size].copy()
        self.bc_prob    = self.bc_prob[:self.crt_size].copy()

        print(self.note.shape, self.next_note.shape, self.state.shape, self.action.shape, self.next_state.shape, self.reward.shape, self.done.shape)
        print(f"{self.target_data} Replay Buffer loaded with {self.crt_size} elements.") 
        return self
    
    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)

        return(
            torch.FloatTensor(self.note[ind]).to(self.device),
            torch.FloatTensor(self.next_note[ind]).to(self.device),
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device),
            torch.FloatTensor(self.bc_prob[ind]).to(self.device),
            torch.FloatTensor(self.note_bg_only[ind]).to(self.device),
            torch.FloatTensor(self.next_note_bg_only[ind]).to(self.device)
        )
    

    #def add(self, notes, next_notes, states, action, next_state, reward, done):
    #    self.note[self.ptr] = notes
    #    self.next_note[self.ptr] = next_notes
    #    self.state[self.ptr] = states
    #    self.next_state[self.ptr] = next_state
    #    self.action[self.ptr] = action
    #    self.reward[self.ptr] = reward
    #    self.done[self.ptr] = done

    #    self.ptr = (self.ptr + 1) % self.max_size
    #    self.crt_size = min(self.crt_size + 1, self.max_size)


    #def save(self, only_test_set = False):
    #    if only_test_set:
    #        flag = "test"
    #    else:
    #        flag = "train_val"
    #    np.save(f"{self.buffer_path}{flag}_note.npy", self.note[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_next_note.npy", self.next_note[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_state.npy", self.state[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_action.npy", self.action[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_next_state.npy", self.next_state[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_reward.npy", self.reward[:self.crt_size])
    #    np.save(f"{self.buffer_path}{flag}_done.npy", self.done[:self.crt_size])


    
        