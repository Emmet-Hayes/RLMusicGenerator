import numpy as np
import os

from MIDIHyperparameters import EPSILON, GAMMA, PITCH_COUNT, CHORD_TYPE_COUNT, DURATION_COUNT, CLIP_LENGTH
from MIDIUtility import getKeyModifier, getScaleDegrees, getChordModifiersByChordType


class MIDIEnvironment:
    def __init__(self, sequence_states, scale, key):
        self.loadStateSpace()
        self.sequence_states = sequence_states
        self.loadScaleFromSequence()
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.currentTime = 0
        self.stepsTaken = 0
        self.loadQvalues()
        self.loadCvalues()
        self.loadPolicy()
        self.loadRewards()
        self.episodes = dict({'State': [], 'Action': [], 'probs' : [], 'Reward' : [None]})
        if scale != None and key != None:
            self.scale_mode = True
            self.scale = scale
            self.key = key
        else:
            self.scale_mode = False
        self.correct_note_reward_hf_mod = 0
        self.correct_note_reward = 0
        self.correct_timing_reward_hf_mod = 0
        self.correct_timing_reward = 0
        self.correct_key_reward_hf_mod = 0
        self.correct_key_reward = 0
        self.chord_history = []

    def reset(self):
        self.episodes = dict({'State': [], 'Action': [], 'probs' : [], 'Reward' : [None]})
        self.currentTime = 0.0
        self.stepsTaken = 0
        self.chord_history = []

    def start(self):
        state = [0, 0, 0, 0]
        return state

    def step(self, state, action):
        self.episodes['Action'].append(action)

        next_state =  self.getNextState(state, action)

        if next_state[3] >= 8 * CLIP_LENGTH: # if were now at the end of the clip
            reward = 1
            return reward, next_state, True

        reward = self.calculateReward(next_state)

        self.episodes['Reward'].append(reward)
        self.episodes['State'].append(next_state)
        self.stepsTaken += 1

        return reward, next_state, False

    def getNextState(self, state, action):
        next_state = [-1, -1, -1, -1]
        next_state[0] = action[0] # pitch
        next_state[1] = action[1] # chord type
        next_state[2] = action[2] # duration
        self.currentTime += action[2] + 1 # add it to the timer
        next_state[3] = int(self.currentTime)

        return next_state

    def calculateReward(self, next_state):
        reward = -1

        # if we use a scale, reward based on playing pleasing scale degrees
        if self.scale_mode == True:

            keyModifier = getKeyModifier(self.key)
            scaleDegrees = getScaleDegrees(self.scale)
            chordMods = getChordModifiersByChordType(next_state[1])

            for chordMod in chordMods:
                if next_state[0] + chordMod % 12 == (scaleDegrees[0] + keyModifier + 1) % 12:
                    reward += 15 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[1] + keyModifier + 1) % 12:
                    reward += 3 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[2] + keyModifier + 1) % 12:
                    reward += 10 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[3] + keyModifier + 1) % 12:
                    reward += 7 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[4] + keyModifier + 1) % 12:
                    reward += 12 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[5] + keyModifier + 1) % 12:
                    reward += 7 + self.correct_key_reward_hf_mod
                elif next_state[0] + chordMod % 12 == (scaleDegrees[5] + keyModifier + 1) % 12:
                    reward += 1 + self.correct_key_reward_hf_mod
                self.correct_key_reward += reward
        else:
            # else reward notes that may be in the wrong octave, but are in the right key (if there is one)
            chordMods = getChordModifiersByChordType(next_state[1])

            for chordMod in chordMods:
                if (next_state[0] + chordMod % 12) in self.key_sequence:
                    # reward decay for chord and starting note repetition
                    current_chord = (next_state[0], next_state[1])
                    repeat_count = self.chord_history.count(current_chord)
                    if repeat_count > 0:
                        decay_factor = 0.5 ** repeat_count
                        chord_repeat_reward = 6 * decay_factor
                        reward += chord_repeat_reward
                    else:
                        reward += 6 + self.correct_key_reward_hf_mod
                    self.correct_key_reward += reward

        # reward well for playing the right timing of a note in the sequence
        if next_state[1] in self.sequence_states[1]:
            reward += 2 + (2 * self.correct_timing_reward_hf_mod)
            self.correct_timing_reward += reward

        return reward


    def loadScaleFromSequence(self):
        self.key_sequence = []
        for note in self.sequence_states:
            self.key_sequence.append(note[0] % 12)
        self.key_sequence = set(self.key_sequence)

    def loadStateSpace(self):
        self.state_space = []
        for i in range(PITCH_COUNT):
            for j in range(DURATION_COUNT):
                for k in range(DURATION_COUNT * CLIP_LENGTH):
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
            self.policy = np.zeros((PITCH_COUNT, CHORD_TYPE_COUNT, DURATION_COUNT, DURATION_COUNT * CLIP_LENGTH), dtype = 'int')
        else:
            self.policy = np.load('Policy.npy')

    def saveCvalues(self, filename = 'Cvalues.npy'):
        np.save(filename, self.Cvalues)

    def loadCvalues(self):
        if not os.path.exists('Cvalues.npy'):
            self.Cvalues = np.zeros((PITCH_COUNT, CHORD_TYPE_COUNT, DURATION_COUNT, DURATION_COUNT * CLIP_LENGTH, PITCH_COUNT * CHORD_TYPE_COUNT * DURATION_COUNT))
        else:
            self.Cvalues = np.load('Cvalues.npy')

    def saveQvalues(self, filename = 'Qvalues.npy'):
        np.save(filename, self.Qvalues)

    def loadQvalues(self):
        if not os.path.exists('Qvalues.npy'):
            self.Qvalues = np.random.rand(PITCH_COUNT, CHORD_TYPE_COUNT, DURATION_COUNT, DURATION_COUNT * CLIP_LENGTH, PITCH_COUNT * CHORD_TYPE_COUNT * DURATION_COUNT) * 400 - 500
        else:
            self.Qvalues = np.load('Qvalues.npy')
