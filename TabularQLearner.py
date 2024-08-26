import numpy as np

class TabularQLearner:

    def __init__ (self, states = 100, actions = 4, alpha = 0.2, gamma = 0.9,
                  exploration = 'eps', epsilon = 0.98, epsilon_decay = 0.999, dyna = 0):

        # Store all the parameters as attributes (instance variables).
        # Initialize any data structures you need.
        self.states = states
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.dyna = dyna # number of hallucinations per real experience

        self.q_table = np.random.random((states, actions))
        self.prev_state = 0
        self.prev_action = 0
        self.history = []


    def train (self, s, r, old_s = None, old_a = None, eval_function = None,
               select_function = None):

        # Receive new state s and new reward r. Update Q-table and return selected action.
        #
        # Consider: The Q-update requires a complete <s, a, s', r> tuple.
        # In this part of the project, the optional parameters are not used.
        # How will you know the previous state and action?

        # simply keep a history of all the real-world experience tuples you have ever seen (including duplicates), 
        # and when you need to hallucinate tuples, sample from that.


         if self.prev_state is not None and self.prev_action is not None:
            experience = (self.prev_state, self.prev_action, s, r)
            self.history.append(experience)
            old_q = self.q_table[self.prev_state, self.prev_action]
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.actions)
                self.epsilon *= self.epsilon_decay
            else:
                action = np.argmax(self.q_table[s, :])
            new_q = (1 - self.alpha)*old_q + self.alpha * (r + self.gamma * np.max(self.q_table[s,action]))
            self.q_table[self.prev_state, self.prev_action] = new_q
            for hallucination in range(self.dyna):
                # randomly sample from the history of experiences
                random_i = np.random.randint(0,len(self.history))
                sh, ah, sph, rh = self.history[random_i]
                old_q = self.q_table[sh, ah]
                new_q = (1 - self.alpha)*old_q + self.alpha * (rh + self.gamma * np.max(self.q_table[sph]))
                self.q_table[sh, ah] = new_q


            self.prev_state = s
            self.prev_action = action

            return action




    def test (self, s, select_function = None, allow_random = False):

        # Receive new state s. Do NOT update Q-table, but still return selected action.
        #
        # This method is called for TWO reasons: (1) to use the policy after learning is finished, and
        # (2) when there is no previous state or action (and hence no Q-update to perform).
        #
        # You sometimes will, and sometimes won't, want to allow random actions...
        
        action = np.argmax(self.q_table[s, :])
        self.prev_state = s
        self.prev_action = action

        return action

        


    def getStateValues (self):

        # Return the max Q value for every state as a 1-D numpy array.
        # This is needed for robot_env to draw its plots.

        return np.max(self.q_table, axis=1)
