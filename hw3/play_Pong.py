import gym
import dqn
import torch
from torch import nn
from dqn_utils import PiecewiseSchedule, get_wrapper_by_name
from atari_wrappers import wrap_deepmind
from dqn_utils import LinearSchedule, ReplayBuffer, get_wrapper_by_name
import time
def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_normal_(m.weight)
    if hasattr(m, 'bias'):
        nn.init.constant_(m.bias, 0)
# 想要load 模型，
class DQN(nn.Module): # for atari
    def __init__(self, in_channels, num_actions):
        # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        super(DQN, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512),
            nn.ReLU(True),
            # Atari环境下动作都是离散的，这里拟合Q值的思路不是把state和action一起输入，最后输出一个值作为拟合的Q值，而是
            # state作输入，输出num_action个Q值，作为不同动作的Q值，显然这样设计无法应用到连续动作空间中。
            nn.Linear(in_features=512, out_features=num_actions),
        )

        self.apply(weights_init)

    def forward(self, obs):
        out = obs.float() / 255 # convert 8-bits RGB color to float in [0, 1]
        # permute：将tensor的维度换位。
        # ob_shape:
        #   self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        # frames：把多个最近的frame当做一个observation
        out = out.permute(0, 3, 1, 2) # reshape to [batch_size, img_c * frames, img_h, img_w]
        out = self.convnet(out)
        out = out.view(out.size(0), -1) # flatten feature maps to a big vector
        out = self.classifier(out)
        return out

# Choose Atari games.
env_name = 'PongNoFrameskip-v4'
env = gym.make(env_name)
env = wrap_deepmind(env)
q_net = torch.load("model.pkl").cuda()
replay_buffer = ReplayBuffer(1000000, 4, lander=False)
ob = env.reset()

while True:
    env.render()
    time.sleep(0.02)
    replay_idx = replay_buffer.store_frame(ob)
    feed_frame = torch.from_numpy(replay_buffer.encode_recent_observation()[None]).cuda()
    q_values = q_net(feed_frame)
    action = torch.argmax(q_values, dim=1)[0].item()
    ob, reward, done, _ = env.step(action)
    replay_buffer.store_effect(replay_idx, action, reward, done)
    if done:
        ob = env.reset()
