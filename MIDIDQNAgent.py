import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from MIDIDQN import DQN


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = self.state_to_one_hot(state)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = self.state_to_one_hot(state)
            next_state = self.state_to_one_hot(next_state)
            target = reward
            if not done:
                next_state = torch.from_numpy(next_state).float().unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            state = torch.from_numpy(state).float().unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def state_to_one_hot(self, state_index):
        state_vector = np.zeros(self.state_size)
        if state_index < self.state_size:
            state_vector[state_index] = 1
        return state_vector

    def saveModel(self):
        torch.save(self.model.state_dict(), 'dqn_model.pth')
