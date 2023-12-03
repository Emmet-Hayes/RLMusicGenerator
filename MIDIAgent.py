import numpy as np
import matplotlib.pyplot as plt

from MIDIHyperparameters import PITCH_COUNT, DURATION_COUNT, CLIP_LENGTH


class MIDIAgent:
    def __init__(self, env):
        self.loadActionSpace()
        self.env = env
        for i in range(PITCH_COUNT):
            for j in range(DURATION_COUNT): # any time in 10 s subdivided by 0.125 sec increments
                for k in range(CLIP_LENGTH * 8): # 8th, qtr, half, whole
                    self.env.policy[i, j, k] = np.argmax(self.env.Qvalues[i, j, k])

    def loadActionSpace(self):
        self.action_space = []
        for i in range(PITCH_COUNT):
            for j in range(DURATION_COUNT):
                self.action_space.append([i, j])

    def possible_actions(self):
        possible_actions = []
        for i, x in enumerate(self.action_space):
            possible_actions.append(i)

        possible_actions = np.array(possible_actions)
        return possible_actions

    def mapAction1D(self, action):
        for i, x in enumerate(self.action_space):
            if list(action) == list(x):
                return i

    def getAction(self, state, policy):
        return self.action_space[(policy(state, self.possible_actions()))]

    def mcControl(self):
        self.env.reset()
        state = self.env.start()
        self.env.episodes['State'].append(state)
        reward = -1
        done = False

        while not done:
            action = self.getAction(state, self.generateActionFromBehaviorPolicy)
            reward, state, done = self.env.step(state, action)

        G = 0 # returns averaged each episode
        W = 1 # importance sampling
        T = self.env.stepsTaken

        # loop backwards through the time steps
        for t in range(T - 1, -1, -1):
            G = self.env.gamma * G + self.env.episodes['Reward'][t + 1]
            S_t = tuple(self.env.episodes['State'][t]) # s a s' r tuple
            A_t = self.mapAction1D(self.env.episodes['Action'][t]) # (3, 3) flattened to 9 actions
            SA = S_t + (A_t,)

            # print("SA: " + str(SA))

            self.env.Cvalues[SA] += W
            self.env.Qvalues[SA] += (W * (G - self.env.Qvalues[SA])) / self.env.Cvalues[SA]
            self.env.policy[S_t] = np.argmax(self.env.Qvalues[S_t])

            if A_t != self.env.policy[S_t]:
                break

            W /= self.env.episodes['probs'][t]

    # off-policy Double-Q-learning control algorithm based on Sutton and Barto definition
    def qControl(self):
        self.env.reset()
        state = self.env.start()
        done = False

        while not done:
            action = self.getAction(state, self.generateActionFromBehaviorPolicy)
            reward, next_state, done = self.env.step(state, action)

            A_t = self.mapAction1D(action)
            SA = tuple(state) + (A_t,)

            #print(str(SA))

            # Q-learning update
            if np.random.rand() < 0.5: # randomly choose which q-value table to update
                if not done:
                    max_next_action = np.argmax(self.env.Qvalues[SA])
                    self.env.Qvalues[SA] += self.env.alpha * (reward + self.env.gamma * max_next_action - self.env.Qvalues[SA])
                else:
                    self.env.Qvalues[SA] += self.env.alpha * (reward - self.env.Qvalues[SA])
            else:
                if not done:
                    max_next_action = np.argmax(self.env.Qvalues2[SA])
                    self.env.Qvalues2[SA] += self.env.alpha * (reward + self.env.gamma * max_next_action - self.env.Qvalues2[SA])
                else:
                    self.env.Qvalues2[SA] += self.env.alpha * (reward - self.env.Qvalues2[SA])

            # Update the policy for the state using the average of the two Q-value tables
            self.env.policy[tuple(state)] = np.argmax((self.env.Qvalues[tuple(state)] + self.env.Qvalues2[tuple(state)]) / 2)

            # Move to the next state
            state = next_state

        # Update the alpha to encourage convergent behavior
        self.env.alpha *= 0.9999 # alpha decay factor

    def evaluateTargetPolicy(self):
        self.env.reset()
        state = self.env.start()
        self.env.episodes['State'].append(state)
        reward = -1
        done = False

        while not done:
            action = self.getAction(state, self.generateActionFromTargetPolicy)
            reward, state, done = self.env.step(state, action)

        self.env.rewards.append(sum(self.env.episodes['Reward'][1:]))

    def outputStatesFromTargetPolicyRun(self):
        self.env.reset()
        state = self.env.start()
        reward = -1
        done = False

        states = []
        while not done:
            action = self.getAction(state, self.generateActionFromTargetPolicy)
            reward, state, done = self.env.step(state, action)
            states.append(state)

        return states

    def calculateBehaviorProbabilities(self, state, action, possible_actions):
        best_action = self.env.policy[tuple(state)]
        num_actions = len(possible_actions)

        if best_action in possible_actions:
            if action == best_action:
                prob = 1 - self.env.epsilon + self.env.epsilon / num_actions
            else:
                prob = self.env.epsilon / num_actions
        else:
            prob = 1 / num_actions

        self.env.episodes['probs'].append(prob)

    def generateActionFromTargetPolicy(self, state, possible_actions):
        if self.env.policy[tuple(state)] in possible_actions:
            action = self.env.policy[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        return action

    def generateActionFromBehaviorPolicy(self, state, possible_actions):
        # print('state: ' + str(state))
        if np.random.rand() > self.env.epsilon and self.env.policy[tuple(state)] in possible_actions:
            action = self.env.policy[tuple(state)]
        else:
            action = np.random.choice(possible_actions)

        self.calculateBehaviorProbabilities(state, action, possible_actions)

        return action

    def plotRewards(self, iterCount, plot_name = "Monte Carlo Performance"):
        ax, fig = plt.subplots(figsize=(30, 15))
        x = np.arange(1, len(self.env.rewards) + 1)
        plt.plot(x * 10, self.env.rewards, linewidth=0.5, color = '#8F719C')
        plt.xlabel('Episode number', size = 24)
        plt.ylabel('Reward', size = 24)
        plt.title(plot_name, size = 36)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        plt.savefig('RewardGraph' + str(len(self.env.rewards) * 10) + '.png')
        plt.close()

    def saveTrackData(self):
        self.env.saveQvalues()
        self.env.saveCvalues()
        self.env.savePolicy()
        self.env.saveRewards()
