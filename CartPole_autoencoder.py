import torch
import gym
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T

from PIL import Image


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.dconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
        self.dconv2 = nn.ConvTranspose2d(16,3,kernel_size=5, stride=2)

    def forward(self, x):
        x = nn.ReLU(self.conv1(x))
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))

        x = nn.ReLU(self.dconv1(x))
        x = self.dconv2(x)

        return x


env = gym.make('CartPole-v0').unwrapped
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600


def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART


def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)



env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

# sample from CartPole
print('collecting images...')

Episode = 100
sample = []
for i_episode in range(Episode):
    env.reset()
    while True:
        sample.append(get_screen())
        s_, reward, done, _ = env.step(i_episode % 2)

        if done:
            break

print(len(sample))
