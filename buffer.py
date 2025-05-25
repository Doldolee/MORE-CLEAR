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
                 buffer_size = 200000,
                 device = 'cuda'
                 ):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device
        self.target_data = target_data
        # self.buffer_path = f"./dataset/{self.target_data}/buffer_clinical_bert/"
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

    
    def load_original_dataset(self, size=-1, only_test_set = False):
        if only_test_set:
            flag = "test"
        else:
            flag = "train_val"

        reward_buffer = np.load(f"{self.buffer_path}{flag}_reward.npy")
      
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


    
        