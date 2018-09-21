import torch as t
from torch import nn
from torch import  optim
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import os
import gym

def loader(path):
    all = np.load(path)
    X = t.Tensor(all['arr_0'])
    y = t.Tensor(all['arr_1'])
    return X, y.squeeze(1)

class ObAct(data.Dataset):
    def __init__(self, path):
        all = np.load(path)
        self.Ob = t.Tensor(all['arr_0'])
        self.Act = t.Tensor(all['arr_1']).squeeze(1)
    def __getitem__(self, index):
        return self.Ob[index, :], self.Act[index, :]
    def __len__(self):
        return self.Ob.size(0)

NUM_OB = -1
NUM_ACT = -1
HIDDEN1 = 128
HIDDEN2 = 256
HIDDEN3 = 64
MAX_STEPS = 5000


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('num_ob', type=int)
    parser.add_argument('num_act', type=int)
    args = parser.parse_args()
    
    PROFECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    train_file = os.path.join(PROFECT_ROOT, 'data', args.envname+'.train.npz')
    NUM_OB = args.num_ob
    NUM_ACT = args.num_act
    
    # X_raw, y_raw = loader(train_file)
    # print(X_raw.size(), y_raw.size())

    obAct = ObAct(train_file)
    dataloader = DataLoader(obAct, batch_size=200)

    # _ = 3
    # for batch_obs, batch_acts in dataloader:
    #     if _ >= 0:
    #         print(batch_obs.size(), batch_acts.size())
    #         _ -= 1

    # 指定requeire_grad——不需要，Module里的参数已经设好了吧 指定train和eval——调用net.train()/net.eval()
    net = nn.Sequential(
        nn.Linear(NUM_OB, HIDDEN1),
        nn.ReLU(),
        nn.Linear(HIDDEN1, HIDDEN2),
        nn.ReLU(),
        nn.Linear(HIDDEN2, HIDDEN3),
        nn.ReLU(),
        nn.Linear(HIDDEN3, NUM_ACT)
    )

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad()

    it = iter(dataloader)
    batch_obs, batch_acts = it.next()
    print(args.envname, "batch_obs", batch_obs.size())
    print(args.envname, "batch_acts", batch_acts.size())
    
    i = 0
    epoch = 10
    while epoch >= 0:
        for batch_obs, batch_acts in dataloader:
            out_acts = net(batch_obs)
            loss = loss_fn(out_acts, batch_acts)
            if i % 100 == 0:
                print(epoch, i, loss)
            i += 1
            loss.backward()
            optimizer.step()
        epoch -= 1

    env = gym.make(args.envname)
    rollouts = 20
    returns = []
    for i in range(rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = t.Tensor(obs).unsqueeze(0)
            action = net(obs).detach().numpy()
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            env.render()
            if steps >= MAX_STEPS:
                break
        returns.append(totalr)

    log_file = os.path.join(PROFECT_ROOT, 'data', "BClog.txt")

    import sys
    sys.stdout = open(log_file, 'a')

    print('returns for', args.envname)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == '__main__':
    main()