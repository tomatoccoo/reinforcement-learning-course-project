import gym
import numpy as np
import random
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.functional as F
#matplotlib inline
env = gym.make('FrozenLake-v0')

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.W = nn.Linear(16, 4, bias=False)
        self.W.weight.data.uniform_(0, 0.01)
    def forward(self, x):

        actions_value = self.W(x)
        return actions_value


dqn = Net()
# Set learning parameters
y = .99
e = 0.1
num_episodes = 20000
#create lists to contain total rewards and steps per episode
jList = []
rList = []

succeed_episode = 0
optimizer = torch.optim.SGD(dqn.parameters(), lr=0.1)

for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Network
    while True:
        j+=1
        #Choose an action by greedily (with e chance of random action) from the Q-network

        allQ = dqn(torch.FloatTensor(np.identity(16)[s:s+1]))
        a = allQ.max(1)[1].numpy()
        if np.random.rand(1) < e:
            a[0] = env.action_space.sample()
        #Get new state and reward from environment
        s1,r,d,_ = env.step(a[0])
        #Obtain the Q' values by feeding the new state through our network
        Q1 = dqn(torch.FloatTensor(np.identity(16)[s1:s1+1]).squeeze(0))
        #Obtain maxQ' and set our target value for chosen action.
        maxQ1 = Q1.max()
        targetQ = allQ.clone()
        targetQ[0,a[0]] = r + y*maxQ1
        #Train our network using target and predicted Q values
        loss = ((allQ - targetQ.detach())**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rAll += r
        s = s1
        if d:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            if r > 0:
                succeed_episode += 1

            if i % 100 == 1:
                print('EP: {:d} succeed rate {:4f}'.format(i, succeed_episode / 100))
                succeed_episode = 0

        if d:
            break

    jList.append(j)
    rList.append(rAll)
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
#Percent of succesful episodes: 0.442%
plt.plot(rList)