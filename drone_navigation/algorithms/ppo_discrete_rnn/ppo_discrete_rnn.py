import torch
import torch.nn as nn
import numpy as np
import statistics
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from torch.distributions import Categorical

# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer

class Actor_Critic_RNN(nn.Module):
    def __init__(self, args):
        super(Actor_Critic_RNN, self).__init__()
        self.use_gru = args.use_gru
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]

        # actor network
        self.actor_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            self.actor_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            self.actor_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.actor_fc2 = nn.Linear(args.hidden_dim, args.action_dim)

        # critic network
        self.critic_fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        if args.use_gru:
            self.critic_rnn = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True)
        else:
            self.critic_rnn = nn.LSTM(args.hidden_dim, args.hidden_dim, batch_first=True)
        self.critic_fc2 = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.actor_fc1)
            orthogonal_init(self.actor_rnn)
            orthogonal_init(self.actor_fc2, gain=0.01)
            orthogonal_init(self.critic_fc1)
            orthogonal_init(self.critic_rnn)
            orthogonal_init(self.critic_fc2)

    def actor(self, s):
        s = self.activate_func(self.actor_fc1(s))
        output, self.actor_rnn_hidden = self.actor_rnn(s, self.actor_rnn_hidden)
        return self.actor_fc2(output)

    def critic(self, s):
        s = self.activate_func(self.critic_fc1(s))
        output, self.critic_rnn_hidden = self.critic_rnn(s, self.critic_rnn_hidden)
        return self.critic_fc2(output)

class PPO_discrete_RNN:
    def __init__(self, args):
        self.batch_size       = args.batch_size
        self.mini_batch_size  = args.mini_batch_size
        self.max_train_steps  = args.max_train_steps
        self.lr               = args.lr
        self.gamma            = args.gamma
        self.lamda            = args.lamda
        self.epsilon          = args.epsilon
        self.K_epochs         = args.K_epochs
        self.entropy_coef     = args.entropy_coef
        self.set_adam_eps     = args.set_adam_eps
        self.use_grad_clip    = args.use_grad_clip
        self.use_lr_decay     = args.use_lr_decay
        self.device           = args.device

        self.ac = Actor_Critic_RNN(args).to(self.device)
        if self.set_adam_eps:
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=self.lr)

    def reset_rnn_hidden(self):
        self.ac.actor_rnn_hidden = None
        self.ac.critic_rnn_hidden = None

    def choose_action(self, s, evaluate=False):
        s_tensor = torch.tensor(s, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logit = self.ac.actor(s_tensor)
            dist = Categorical(logits=logit)
            if evaluate:
                return torch.argmax(logit, dim=-1).item(), None
            a = dist.sample()
            return a.item(), dist.log_prob(a).item()

    def get_value(self, s):
        s_tensor = torch.tensor(s, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.ac.critic(s_tensor).item()

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()
        # move batch tensors to device
        for k in ['s', 'a', 'a_logprob', 'adv', 'v_target', 'active']:
            dtype = torch.float if k in ['s','a_logprob','adv','v_target'] else torch.long
            batch[k] = torch.as_tensor(batch[k], device=self.device, dtype=dtype)

        # 用来存所有 mini-batch 的指标
        metrics = {
            'loss/policy_loss':    [],
            'loss/value_loss':     [],
            'loss/entropy':        [],
            'loss/clip_fraction':  [],
            'loss/approx_kl':      []
        }

        global_step = 0
        for epoch in range(self.K_epochs):
            for idx in BatchSampler(SequentialSampler(range(self.batch_size)),
                                    self.mini_batch_size, False):
                self.reset_rnn_hidden()

                s_batch = batch['s'][idx]                                     # [M, ...]
                logits  = self.ac.actor(s_batch)                              # [M, n_actions]
                values  = self.ac.critic(s_batch).squeeze(-1)                 # [M]

                dist    = Categorical(logits=logits)
                entropy = dist.entropy().mean()                              # scalar

                logprobs_new = dist.log_prob(batch['a'][idx])               # [M]
                logprobs_old = batch['a_logprob'][idx]                      # [M]
                ratios = torch.exp(logprobs_new - logprobs_old)              # [M]

                # —— Policy loss —— 
                surr1 = ratios * batch['adv'][idx]
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * batch['adv'][idx]
                policy_loss_per_step = -torch.min(surr1, surr2)              # [M]

                active_mask = batch['active'][idx].float()
                policy_loss = (policy_loss_per_step * active_mask).sum() / active_mask.sum()

                # —— Value loss —— 
                value_loss_per_step = (values - batch['v_target'][idx])**2
                value_loss = (value_loss_per_step * active_mask).sum() / active_mask.sum()

                # —— Clip fraction & approx KL —— 
                with torch.no_grad():
                    clip_frac = ((ratios > 1+self.epsilon) | (ratios < 1-self.epsilon)).float()
                    clip_fraction = (clip_frac * active_mask).sum() / active_mask.sum()

                    approx_kl = (logprobs_old - logprobs_new)            # [M]
                    approx_kl = (approx_kl * active_mask).sum() / active_mask.sum()

                # —— 反向＋更新 —— 
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.optimizer.step()

                # —— 收集到 metrics dict —— 
                metrics['loss/policy_loss'].append(policy_loss.item())
                metrics['loss/value_loss'].append(value_loss.item())
                metrics['loss/entropy'].append(entropy.item())
                metrics['loss/clip_fraction'].append(clip_fraction.item())
                metrics['approx_kl'].append(approx_kl.item())

                global_step += 1

        # 可选：在这里做 learning‐rate decay
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        avg_metrics = {
            k: statistics.mean(v_list) if v_list else 0.0
            for k, v_list in metrics.items()
        }
        return avg_metrics

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps/self.max_train_steps) + 0.1 * self.lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr_now
