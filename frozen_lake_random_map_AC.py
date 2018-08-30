import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

#env = gym.make('CartPole-v0')
#env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

IS_SLIPPERY = False  # is the lake slippery
use_random_map = False
HOLE_NUM = 1  # the number of holes



def random_map(holes_num=2):
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



env = FrozenLakeEnv(desc=random_map(HOLE_NUM), is_slippery=IS_SLIPPERY)
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = 16 + 16 + 16  # 'F', 'H', 'where is the Agent'
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(),
                              int) else env.action_space.sample().shape  # to confirm the shape





episode_durations = []


def plot_durations():
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode(1000)')
    plt.ylabel('Mean reward')
    # plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 1000:
        means = durations_t.unfold(0, 1000, 1000).mean(1).view(-1)
        means = torch.cat((torch.zeros(1), means))

        A, = plt.plot(means.numpy(), label='Mean reward of 1000 Episode ')

    if len(durations_t) >= 10000:
        means = durations_t.unfold(0, 10000, 1000).mean(1).view(-1)
        means = torch.cat((torch.zeros(1), means))

        B, = plt.plot(means.numpy(), label='Mean reward of 10000 Episode')

        # plot_data = means.numpy()[range(0,len(means), len(means) // 100)]

        legend = plt.legend(handles=[A, B])

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

    plt.savefig("./result/actor_critic_one_map_slippery.png")
    plt.close()


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(N_STATES, 512)
        self.affine1.weight.data.uniform_(0, 0.01)  # initialization

        self.affine2 = nn.Linear(512, 512)
        self.affine2.weight.data.uniform_(0, 0.01)  # initialization

        self.action_head = nn.Linear(512, 4)
        self.action_head.weight.data.uniform_(0, 0.01)  # initialization

        self.value_head = nn.Linear(512, 1)
        self.value_head.weight.data.uniform_(0, 0.01)  # initialization

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)

        return F.softmax(action_scores, dim=-1), state_values


model = Policy().to(device)
optimizer = optim.SGD(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()

    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)

    #rewards = (rewards - rewards.mean()) / (rewards.std() + eps)

    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    new_map = ["SFFF", "FHFH", "FFFH", "HFFG"]
    env = FrozenLakeEnv(desc=new_map, is_slippery=IS_SLIPPERY)
    env = env.unwrapped
    succeed_episode = 0

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
            a = select_action(state)

            pos_next, r, done, info = env.step(a)
            ep_r += r
            #state_next = encode_state(new_map, pos_next)

            if args.render:
                env.render()
            model.rewards.append(r)

            if done:
                break


        finish_episode()

        episode_durations.append(ep_r)

        if ep_r > 0:
            # EPSILON = 1 - 1. / ((i_episode / 500) + 10)
            succeed_episode += 1

        if i_episode % 1000 == 1:
            print('EP: {:d} succeed rate {:4f}'.format(i_episode, succeed_episode / 1000))
            succeed_episode = 0

        if i_episode % 5000 == 1:
            plot_durations()


torch.save(Policy, 'AC_classical.pkl')

if __name__ == '__main__':
    main()
