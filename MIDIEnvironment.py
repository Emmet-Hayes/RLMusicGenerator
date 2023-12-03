import numpy as np
import os

from MIDIHyperparameters import EPSILON, GAMMA, ALPHA, PITCH_COUNT, DURATION_COUNT, CLIP_LENGTH


class MIDIEnvironment:
    def __init__(self, sequence_states):
        self.loadStateSpace()
        self.sequence_states = sequence_states
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.alpha = ALPHA
        self.currentTime = 0
        self.stepsTaken = 0
        self.loadQvalues()
        self.loadCvalues()
        self.loadPolicy()
        self.loadRewards()
        self.episodes = dict({'State': [], 'Action': [], 'probs' : [], 'Reward' : [None]})

    def reset(self):
        self.episodes = dict({'State': [], 'Action': [], 'probs' : [], 'Reward' : [None]})
        self.currentTime = 0.0
        self.stepsTaken = 0

    def start(self):
        state = [0, 0, 0]
        return state

    def step(self, state, action):
        self.episodes['Action'].append(action)
        reward = -1
        next_state =  self.getNextState(state, action)

        if next_state[2] >= 8 * CLIP_LENGTH: # if were now at the end of the clip
            reward = 1
            return reward, next_state, True

        if next_state[0] == 0: # if this is a resting beat
            reward = 0 # neutral reward

        # if the next state is in the rewardable sequence (not strictly at the same instant), we rejoice
        next_state_trimmed = next_state[0:2]
        sequence_states_trimmed = self.sequence_states[0:2]
        if next_state_trimmed in sequence_states_trimmed:
            reward = 5

        if next_state in self.sequence_states:
            reward = 20

        self.episodes['Reward'].append(reward)
        self.episodes['State'].append(next_state)
        self.stepsTaken += 1

        return reward, next_state, False

    def getNextState(self, state, action):
        next_state = state.copy()
        next_state[0] = action[0] # pitch
        next_state[1] = action[1] # duration
        self.currentTime += action[1] + 1 # add it to the timer
        next_state[2] = int(self.currentTime)

        return next_state

    def loadStateSpace(self):
        self.state_space = []
        for i in range(PITCH_COUNT):
            for j in range(DURATION_COUNT):
                for k in range(8 * CLIP_LENGTH):
                    self.state_space.append([i, j, k])

    def saveRewards(self, filename = 'Rewards.npy'):
        self.rewards = np.array(self.rewards)
        np.save(filename, self.rewards)
        self.rewards = list(self.rewards)

    def loadRewards(self):
        if not os.path.exists('Rewards.npy'):
            self.rewards = []
        else:
            self.rewards = list(np.load('Rewards.npy'))

    def savePolicy(self, filename = 'Policy.npy'):
        np.save(filename, self.policy)

    def loadPolicy(self):
        if not os.path.exists('Policy.npy'):
            self.policy = np.zeros((PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH), dtype = 'int')
        else:
            self.policy = np.load('Policy.npy')

    def saveCvalues(self, filename = 'Cvalues.npy'):
        np.save(filename, self.Cvalues)

    def loadCvalues(self):
        if not os.path.exists('Cvalues.npy'):
            self.Cvalues = np.zeros((PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH, PITCH_COUNT * DURATION_COUNT))
        else:
            self.Cvalues = np.load('Cvalues.npy')

    def saveQvalues(self, filename = 'Qvalues.npy'):
        np.save(filename, self.Qvalues)
        np.save(filename, self.Qvalues2)

    def loadQvalues(self):
        if not os.path.exists('Qvalues.npy'):
            self.Qvalues = np.random.rand(PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH, PITCH_COUNT * DURATION_COUNT) * 400 - 500
        else:
            self.Qvalues = np.load('Qvalues.npy')
        if not os.path.exists('Qvalues2.npy'):
            self.Qvalues2 = np.random.rand(PITCH_COUNT, DURATION_COUNT, 8 * CLIP_LENGTH, PITCH_COUNT * DURATION_COUNT) * 400 - 500
        else:
            self.Qvalues2 = np.load('Qvalues2.npy')
