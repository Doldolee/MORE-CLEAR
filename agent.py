from network import BertNetCQL, CQLNet, CQLContextGatedFusionMixerNet

import copy
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math


class BaseQ(ABC):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device,
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=1,
                 tau=0.005,
                 hidden_node=0,
                 activation="-",
                 target_data = "-",
                 **kwargs):
        self.device = device
        self.num_actions = num_actions
        self.discount = discount
        self.tau = tau
        self.target_update_frequency = target_update_frequency
        self.iterations = 0
        self.target_data = target_data
        self.state_dim = state_dim
        self.note_emb_dim = kwargs.get("note_emb_dim")

        self.maybe_update_target = self.polyak_target_update if use_polyak_target_update else self.copy_target_update

        
    @abstractmethod
    def build_network(self, state_dim, num_actions, hidden_node, activation):
        pass

    @abstractmethod
    def action(self, state):

        pass

    @abstractmethod
    def train(self, replay_buffer):
        pass

    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        self.Q_target.load_state_dict(self.Q.state_dict())


class MultimodalContextCQL(BaseQ):
    def __init__(
        self,
        num_actions,
        state_dim,
        device,
        discount=0.99,
        optimizer='Adam',
        optimizer_parameters={},
        use_polyak_target_update=False,
        target_update_frequency=1,
        tau=0.005,
        hidden_node=128,
        activation='relu',
        cql_alpha=2.0,
        **kwargs
    ):
        self.cql_alpha = cql_alpha
        super(MultimodalContextCQL, self).__init__(
            num_actions,
            state_dim,
            device,
            discount,
            optimizer,
            optimizer_parameters,
            use_polyak_target_update,
            target_update_frequency,
            tau,
            hidden_node,
            activation,
            **kwargs
        )
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.max_training_steps = kwargs.get("max_timesteps")
        self.warmup_steps = int(0.1 * self.max_training_steps)
        self.training_step_count = 0

        self.decay_steps = kwargs.get("decay_steps", int(0.2 * self.max_training_steps))
        self.decay_gamma = kwargs.get("decay_gamma", 0.8)

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            elapsed = current_step - self.warmup_steps
            num_decays = elapsed // self.decay_steps
            return self.decay_gamma ** num_decays

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.Q_optimizer,
            lr_lambda=lr_lambda
        )

    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return CQLContextGatedFusionMixerNet(
            state_dim=state_dim,
            num_actions=num_actions,
            hidden_node=hidden_node,
            activation=activation,
            note_emb_dim=self.note_emb_dim
        )

    def action(self, state, note_emb, context_emb):
        with torch.no_grad():
            q = self.Q(
                state.to(self.device),
                note_emb.to(self.device),
                context_emb.to(self.device)
            )
            return int(q.argmax(dim=1))

    def train(self, replay_buffer):
        self.Q.train()

        note_emb, next_note_emb, state, action, next_state, reward, done, _, context_emb, next_context_emb = replay_buffer.sample()

        with torch.no_grad():
            q_next = self.Q_target(
                next_state.to(self.device),
                next_note_emb.to(self.device),
                next_context_emb.to(self.device)
            )
            next_act = q_next.argmax(dim=1, keepdim=True)
            target_q = reward + (1 - done) * self.discount * q_next.gather(1, next_act)

        q_pred = self.Q(
            state.to(self.device),
            note_emb.to(self.device),
            context_emb.to(self.device)
        )
        current_q = q_pred.gather(1, action)
        td_loss = F.mse_loss(current_q, target_q)
        alpha = self.cql_alpha
        lse = torch.logsumexp(q_pred / alpha, dim=1, keepdim=True) * alpha
        data_q = current_q.mean()
        cql_loss = lse.mean() - data_q
        loss = td_loss + cql_loss

        self.Q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
        self.Q_optimizer.step()

        self.lr_scheduler.step()


        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()


class TextCQL(BaseQ):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        device: torch.device,
        discount: float = 0.99,
        optimizer: str = 'Adam',
        optimizer_parameters: dict = {},
        use_polyak_target_update: bool = False,
        target_update_frequency: int = 1,
        tau: float = 0.005,
        hidden_node: int = 256,
        activation: str = 'relu',
        cql_alpha: float = 2.0,
        **kwargs
    ):
        self.cql_alpha = cql_alpha
        super(TextCQL, self).__init__(num_actions, 
                                      state_dim, device,
                                      discount,
                                      optimizer,
                                      optimizer_parameters,
                                      use_polyak_target_update,
                                      target_update_frequency,
                                      tau, 
                                      hidden_node, 
                                      activation, 
                                      **kwargs
                                )
        
        self.Q = self.build_network(num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

        self.max_training_steps = kwargs.get('max_timesteps', 100000)
        self.warmup_steps = int(0.1 * self.max_training_steps)
        def lr_lambda(step: int) -> float:
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            progress = float(step - self.warmup_steps) / float(max(1, self.max_training_steps - self.warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def build_network(self, num_actions: int, hidden_node: int, activation: str):
        return BertNetCQL(num_actions, hidden_node, activation, note_emb_dim=self.note_emb_dim)

    def action(self, note_emb: torch.Tensor) -> int:
        with torch.no_grad():
            q = self.Q(note_emb)
            return int(q.argmax(dim=1).item())

    def train(self, replay_buffer):
        self.Q.train()
        note, next_note, _, action, _, reward, done, _, _, _ = replay_buffer.sample()

        # Bellman error
        current_Q = self.Q(note).gather(1, action)
        with torch.no_grad():
            next_q = self.Q_target(next_note)
            max_next_q, _ = next_q.max(dim=1, keepdim=True)
            target_Q = reward + (1.0 - done) * self.discount * max_next_q
        bellman_loss = F.mse_loss(current_Q, target_Q)

        all_q = self.Q(note)
        lse_q = torch.logsumexp(all_q, dim=1)
        data_q = current_Q.squeeze(1)
        cql_penalty = (lse_q - data_q).mean() * self.alpha

        loss = bellman_loss + cql_penalty

        # optimize + lr step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # target update
        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()

class TabularCQL(BaseQ):
    def __init__(self,
                 num_actions,
                 state_dim,
                 device,
                 discount=0.99,
                 optimizer='Adam',
                 optimizer_parameters={},
                 use_polyak_target_update=False,
                 target_update_frequency=1,
                 tau=0.005,
                 hidden_node=128,
                 activation='relu',
                 cql_alpha=2.0,
                 **kwargs):
        self.cql_alpha = cql_alpha
        super(TabularCQL, self).__init__(num_actions,
                                          state_dim,
                                          device,
                                          discount,
                                          optimizer,
                                          optimizer_parameters,
                                          use_polyak_target_update,
                                          target_update_frequency,
                                          tau,
                                          hidden_node,
                                          activation,
                                          **kwargs)
        self.Q = self.build_network(state_dim, num_actions, hidden_node, activation).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.Q_optimizer = getattr(torch.optim, optimizer)(self.Q.parameters(), **optimizer_parameters)

    def build_network(self, state_dim, num_actions, hidden_node, activation):
        return CQLNet(state_dim, num_actions, hidden_node, activation)

    def action(self, state):
        with torch.no_grad():
            q = self.Q(state.to(self.device))
            return int(q.argmax(dim=1))

    def train(self, replay_buffer):
        self.Q.train()
        _, _, state, action, next_state, reward, done, _, _, _ = replay_buffer.sample()

        # Target Q
        with torch.no_grad():
            q_next = self.Q_target(next_state)
            next_act = q_next.argmax(dim=1, keepdim=True)
            target_q = reward + (1 - done) * self.discount * q_next.gather(1, next_act)

        # Current Q
        q_pred = self.Q(state)
        current_q = q_pred.gather(1, action)

        # TD loss
        td_loss = F.mse_loss(current_q, target_q)

        # CQL conservative loss
        alpha = self.cql_alpha
        lse = torch.logsumexp(q_pred / alpha, dim=1, keepdim=True) * alpha
        data_q = current_q.mean()
        cql_loss = lse.mean() - data_q

        # Total loss
        loss = td_loss + cql_loss

        self.Q_optimizer.zero_grad()
        loss.backward()
        self.Q_optimizer.step()

        self.iterations += 1
        if self.iterations % self.target_update_frequency == 0:
            self.maybe_update_target()
