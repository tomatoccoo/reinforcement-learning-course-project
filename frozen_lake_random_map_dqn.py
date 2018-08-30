import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import random
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


def random_map(holes_num = 2):
    char_list = list('SFFFFFFFFFFFFFFG')
    for i in range(holes_num):
        char_list[random.randint(1, 14)] = 'H'
    my_map = [''.join(char_list[i:i + 4]) for i in [0, 4, 8, 12]]
    return my_map


def encode_state(map, position):
    np_map = np.asarray(map, dtype='c').reshape(16)
    holes = np.where(np_map == b'H', 1, 0)
    forzen = np.where(np_map == b'F', 1, 0)
    position = np.identity(16)[position]
    return np.hstack([holes, forzen, position])


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                     # learning rate
EPSILON = 0.9                # greedy policy
GAMMA = 0.99                    # reward discount
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 2000


IS_SLIPPERY = False        # is the lake slippery
use_random_map = True
HOLE_NUM = 1               # the number of holes

env = FrozenLakeEnv(desc=random_map(HOLE_NUM), is_slippery=IS_SLIPPERY)
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 16 + 16 + 16 # 'F', 'H', 'where is the Agent'
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 128)
        self.fc1.weight.data.uniform_(0, 0.01)   # initialization
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.uniform_(0, 0.01)

        self.out = nn.Linear(64, N_ACTIONS)
        self.out.weight.data.uniform_(0, 0.01)   # initialization

        #self.W = nn.Linear(N_STATES, N_ACTIONS, bias = False)
        #self.W.weight.data.uniform_(0, 0.001)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions_value = self.out(x)

        #actions_value = self.W(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
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

        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

episode_durations = []


plt.figure(1)
def plot_durations():
    plt.close()
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode(1000)')
    plt.ylabel('Mean reward')
    #plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 1000 :
        means = durations_t.unfold(0, 1000, 1000).mean(1).view(-1)
        means = torch.cat((torch.zeros(1), means))

        A, = plt.plot(means.numpy(), label = 'Mean reward of 1000 Episode ')

    if len(durations_t) >= 10000:
        means = durations_t.unfold(0, 10000, 1000).mean(1).view(-1)
        means = torch.cat((torch.zeros(1), means))

        B, = plt.plot(means.numpy(), label = 'Mean reward of 10000 Episode')

        #plot_data = means.numpy()[range(0,len(means), len(means) // 100)]

        legend = plt.legend(handles=[A, B])

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.savefig("./result/dqn_random_map_no_slippery_1_hole.png")



print('\nCollecting experience...')
succeed_episode = 0
new_map = ["SFFF","FHFH","FFFH","HFFG"]
for i_episode in range(1000000):

    if use_random_map and i_episode % 10 == 0:
        env.close()
        new_map = random_map(HOLE_NUM)
        env = FrozenLakeEnv(desc=new_map, is_slippery=IS_SLIPPERY)
        env = env.unwrapped

    pos = env.reset()
    state = encode_state(new_map, pos)
    ep_r = 0
    while True:
        #env.render()

        a = dqn.choose_action(state)

        # take action
        pos_next, r, done, info = env.step(a)

        state_next = encode_state(new_map, pos_next)
        dqn.store_transition(state, a, r, state_next)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                episode_durations.append(ep_r)
                if ep_r > 0:
                    #EPSILON = 1 - 1. / ((i_episode / 500) + 10)
                    succeed_episode += 1

                if i_episode % 1000 == 1:
                    print('EP: {:d} succeed rate {:4f}'.format(i_episode, succeed_episode / 1000))
                    succeed_episode = 0

                if i_episode % 5000 == 1:
                    plot_durations()

        if done:
            break
        state = state_next

torch.save(dqn, 'full_connect_net_params.pkl')

