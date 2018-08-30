"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/
Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# Hyper Parameters
BATCH_SIZE = 16
LR = 0.1                     # learning rate
EPSILON = 0.9                # greedy policy
GAMMA = 0.99                    # reward discount
TARGET_REPLACE_ITER = 50   # target update frequency
MEMORY_CAPACITY = 2000
env = gym.make('FrozenLake-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.n
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.uniform_(0, 0.01)   # initialization
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.uniform_(0, 0.01)   # initialization

        #self.W = nn.Linear(N_STATES, N_ACTIONS, bias = False)
        #self.W.weight.data.uniform_(0, 0.001)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)

        #actions_value = self.W(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print('\nCollecting experience...')
succeed_episode = 0
for i_episode in range(50000):
    s = env.reset()
    ep_r = 0
    while True:
        #env.render()
        one_hot_s = torch.zeros(N_STATES).scatter_(0, torch.LongTensor([s]), 1)
        a = dqn.choose_action(one_hot_s)

        # take action
        s_, r, done, info = env.step(a)
        # modify the reward
        #print(r)

        one_hot_s_ = torch.zeros(N_STATES).scatter_(0, torch.LongTensor([s_]), 1)
        dqn.store_transition(one_hot_s, a, r, one_hot_s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                if ep_r > 0:
                    EPSILON = 1 - 1. / ((i_episode / 500) + 10)
                    succeed_episode += 1

                if i_episode % 1000 == 1:
                    print('EP: {:d} succeed rate {:4f}'.format(i_episode, succeed_episode / 1000))
                    succeed_episode = 0
                    #print('Ep: ', i_episode,
                    #      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_
