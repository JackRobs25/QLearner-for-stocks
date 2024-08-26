import numpy as np
from TabularQLearner import TabularQLearner

class DoubleQLearner:

    def __init__(self, states=100, actions=4, alpha=0.2, gamma=0.9,
                 exploration='eps', epsilon=0.98, epsilon_decay=0.999, dyna=0):
        # Initialize two TabularQLearner objects for double Q learning.
        self.qa = TabularQLearner()
        self.qb = TabularQLearner()
        self.historyA = []
        self.historyB = []
        self.prev_action = None
        self.prev_state = None
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna

    def train(self, s, r, old_s=None, old_a=None, eval_function=None, select_function=None):
        # Train using double Q learning.
        
        # randomly distribute incoming experiences between them
        rand = np.random.random()
        recieve = self.qa if rand <= 0.5 else self.qb
        consult = self.qb if rand <= 0.5 else self.qa
        hist = self.historyA if rand <= 0.5 else self.historyB

        # have to code up the experience tuple and add it to hist
        action = np.random.randint(self.actions)
        if self.prev_action is not None and self.prev_state is not None:
            experience = (self.prev_state, self.prev_action, s, r)
            hist.append(experience)
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.actions)
                self.epsilon *= self.epsilon_decay
            else:
                action = np.argmax(recieve.q_table[s, :])
            old_q = recieve.q_table[self.prev_state, self.prev_action]
            new_q = (1 - self.alpha)*old_q + self.alpha * (r + self.gamma * np.max(consult.q_table[s,action]))
            recieve.q_table[self.prev_state, self.prev_action] = new_q
            for hallucination in range(self.dyna):
                random_i = np.random.randint(0,len(hist))
                sh, ah, sph, rh = hist[random_i]
                old_q = recieve.q_table[sh, ah]
                new_q = (1 - self.alpha)*old_q + self.alpha * (rh + self.gamma * np.max(recieve.q_table[sph]))
                recieve.q_table[sh, ah] = new_q


        self.prev_state = s
        self.prev_action = action

        return action


    def test(self, s, allow_random=False):
        # Double Q test.
        
        q_avg = (self.qa.q_table + self.qb.q_table)/2
        action = np.argmax(q_avg[s, :])
        self.prev_state = s
        self.prev_action = action

        return action

    def getStateValues(self):
        # Still needed for the test environment to work.
        q_avg = (self.qa.q_table + self.qb.q_table)/2
        return np.max(q_avg, axis=1)
