import numpy as np

class LSPIAgent:
    def __init__(self, epsilon, discount, possible_actions, model_path=None):
        self.possible_actions = possible_actions
        self.epsilon = epsilon
        self.discount = discount

        l = 21*len(possible_actions)

        self.A = np.eye(l,l)
        self.b = np.zeros((l,))

        if model_path is None:
            self.w = np.zeros((l,))
        else:
            self.w = np.load(model_path)

    def linear_basis_function(self,s,a):
        if self.purpose == 'play':
            s0 = 1; s1 = s[0]; s2 = s[1]; s3 = s[2]; s4 = s[3]; s5 = s[4]

        elif self.purpose == 'train':
            s0 = np.ones((s.shape[0],)); s1 = s[:,0]; s2 = s[:,1]; s3 = s[:,2]; s4 = s[:,3]; s5 = s[:,4]

        features = [s0, s1, s2, s3, s4, s5, s1*s1, s1*s2, s1*s3, s1*s4, s1*s5,
                    s2*s2, s2*s3, s2*s4, s2*s5, s3*s3, s3*s4, s3*s5, s4*s4, s4*s5, s5*s5]

        f = []
        c = []
        for i in self.possible_actions:
            c.append([a-i] * len(features))
            f.append(features)

        flat_c = np.array([item for sublist in c for item in sublist])
        mask = (flat_c == 0) + 0

        flat_f = np.array([item for sublist in f for item in sublist])

        return flat_f, mask

    def play(self,next_states):
        q = []
        self.purpose = 'play'

        for action in self.possible_actions:
            q.append(self.get_Q(next_states,action))

        next_action = np.array(self.possible_actions)[np.argmax(q,axis=0)]

        return np.array([next_action], np.float64)

    def get_actions(self,next_states):
        q = []

        for action in self.possible_actions:
            next_action = np.array([action] * next_states.shape[0])
            q.append(self.get_Q(next_states,next_action))

        next_actions = np.array(self.possible_actions)[np.argmax(q,axis=0)]

        for i in range(len(next_actions)):
            if np.random.rand(1,1) < self.epsilon:
                next_actions[i] = np.random.choice(possible_actions)

        return np.reshape(next_actions,(next_actions.shape[0],))

    def get_Q(self,states,actions):
        phi,mask = self.linear_basis_function(states,actions)
        Q = np.dot((phi*mask).T, self.w)
        return Q

    def estimate_b(self,states,actions,rewards):
        phi,mask = self.linear_basis_function(states,actions)
        b = np.sum((phi*mask) * rewards, axis=1)
        return b

    def LSPI(self,states,actions,rewards,next_states):
        self.purpose = 'train'
        n_iter = 5
        for _ in range(n_iter):
            w_old = self.w
            A_old = self.A
            b_old = self.b

            phi,mask = self.linear_basis_function(states,actions)
            next_actions = self.get_actions(next_states)
            phi2,mask2 = self.linear_basis_function(next_states,next_actions)

            phi3 = (phi*mask) - self.discount*(phi2*mask2)

            self.A = A_old + np.dot((phi*mask),phi3.T)/(states.shape[0])
            self.b = b_old + self.estimate_b(states,actions,rewards)/(states.shape[0])

            self.w = np.matmul(np.linalg.pinv(self.A),self.b)

            error = np.linalg.norm(self.w - w_old)

            if error < 0.001:
                break

    def sampling(self,env,n):
        states = []
        actions = []
        rewards = []
        next_states = []

        for _ in range(n):
            all_states = []
            all_rewards = []
            all_actions = []
            all_states.append(env.reset())
            done = False
            while not done:
                a = np.random.choice(self.possible_actions)
                s, r, done, _ = env.step(np.array([a], np.float64))
                all_states.append(s)
                all_actions.append(a)
                all_rewards.append(r)
                if done: break

            states.append(all_states[:-1])
            actions.append(all_actions)
            rewards.append(all_rewards)
            next_states.append(all_states[1:])

        states = np.array([item for sublist in states for item in sublist])
        actions = np.array([item for sublist in actions for item in sublist])
        rewards = np.array([item for sublist in rewards for item in sublist])
        next_states = np.array([item for sublist in next_states for item in sublist])

        return states, actions, rewards, next_states
