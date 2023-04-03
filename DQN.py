import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random 
from collections import namedtuple, deque 
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# create the Deep Q network
class DeepQNet(nn.Module):
    def __init__(self, num_state, num_action):
        super(DeepQNet,self).__init__()
        num_nodes = 64
        self.layer1 = nn.Linear(num_state,num_nodes)
        self.layer2 = nn.Linear(num_nodes,num_nodes)
        self.layer3 = nn.Linear(num_nodes,num_action)

    def forward(self, x):
        # mapping state to action values
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        return self.layer3(x)

# Replay buffer for storing experience
class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, *args):
        # add a new memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # sample a random memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # return length of memory
        return len(self.memory)

# DQN agent that interacts with the environment
class Agent():
    def __init__(self, num_state, num_action, alpha, gamma, batch_size, replay):
        self.num_state = num_state
        self.num_action = num_action
        self.batch_size = batch_size
        self.gamma = gamma

        self.local_net = DeepQNet(num_state, num_action).to(device)
        self.target_net = DeepQNet(num_state, num_action).to(device)

        self.optimizer = optim.Adam(self.local_net.parameters(), alpha)

        self.memory = ReplayBuffer(10000)
        if not replay: self.memory = ReplayBuffer(batch_size)

        self.steps = 0

    def choose_action(self, state):
        if not torch.is_tensor(state): state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            return self.local_net(state).max(1)[1].view(1, 1)

    def optimize(self):
        if len(self.memory) < self.batch_size: return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # transitions that do not lead to terminal states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) 

        # values of states, actions, and rewards
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        reward_batch = reward_batch.flatten()

        state_action_values = self.local_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()

        # gradient clipping
        # clip_value = 1
        # for p in self.local_net.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        nn.utils.clip_grad_value_(self.local_net.parameters(), 100)

        self.optimizer.step()

    def train(self, num_episodes, env, reset_num, targetQ):
        rewards = []
        disc_rewards = []

        if not targetQ: reset_num = 1

        for ep in range(num_episodes):
            # print(ep)
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            reward = 0
            disc_reward = 0

            for t in itertools.count():
                action = self.choose_action(state)

                next_state, reward_t, done = env.step(action.item())
                next_state = torch.tensor(next_state, dtype=torch.float, device=device).unsqueeze(0)
                reward_t = torch.tensor([reward_t], dtype=torch.float, device=device).unsqueeze(0)
                reward += reward_t.item()
                disc_reward += (self.gamma ** t) * reward_t.item()
                
                self.memory.add(state, action, next_state, reward_t)
                state = next_state

                self.optimize()

                if done: break

            rewards.append(reward)
            disc_rewards.append(disc_reward)

            if ep % reset_num == 0:
                self.target_net.load_state_dict(self.local_net.state_dict())

        return rewards, disc_rewards



