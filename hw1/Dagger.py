import torch as t
from torch import nn
from torch import  optim
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import os
import gym
import tensorflow as tf
import tf_util
import load_policy

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

def run_env(net, envname, rollouts, policy_fn=None):
    env = gym.make(envname)
    returns = []
    new_obs = []
    new_actions = []
    for i in range(rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = t.Tensor(obs).unsqueeze(0)
            action = net(obs).detach().numpy()
            obs, r, done, _ = env.step(action)
            if policy_fn != None:
                new_obs.append(obs)
                new_act = policy_fn(obs[None, :])
                new_actions.append(new_act)
            totalr += r
            steps += 1
            env.render()
            if steps >= MAX_STEPS:
                break
        returns.append(totalr)
    return new_obs, new_actions, returns

def run_env2(net, envname, rollouts):
    env = gym.make(envname)
    returns = []
    new_obs = []
    for _ in range(rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obs = t.Tensor(obs).unsqueeze(0)
            action = net(obs).detach().numpy()
            obs, r, done, _ = env.step(action)
            new_obs.append(obs)
            totalr += r
            steps += 1
            env.render()
            if steps >= MAX_STEPS:
                break
        returns.append(totalr)
    return new_obs, returns

def main():
    with tf.Session():
        tf_util.initialize()
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('expert_policy_file', type=str)
        parser.add_argument('envname', type=str)
        parser.add_argument('num_ob', type=int)
        parser.add_argument('num_act', type=int)
        args = parser.parse_args()
        
        PROFECT_ROOT = os.path.dirname(os.path.realpath(__file__))
        train_file = os.path.join(PROFECT_ROOT, 'data', args.envname+'.train.npz')    
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')
        NUM_OB = args.num_ob
        NUM_ACT = args.num_act
        
        # X_raw, y_raw = loader(train_file)
        # print(X_raw.size(), y_raw.size())

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

        obAct = ObAct(train_file)
        dataloader = DataLoader(obAct, batch_size=200)

        i = 0
        epoch = 10
        datasets = []
        datasets.append(obAct)
        while epoch >= 0:
            for batch_obs, batch_acts in dataloader:
                out_acts = net(batch_obs)
                loss = loss_fn(out_acts, batch_acts)
                if i % 100 == 0:
                    print(epoch, i, loss)
                i += 1
                loss.backward()
                optimizer.step()
            """
            dagger 先训练了一个epoch，得到当前的net，在环境中跑一遍这个net，相当于可以得到
            **这个net下的obs“转移路径”**，再用expert为这条路径上的所有ob打label
            """
            # new_obs, new_acts, _ = run_env(net, args.envname, 20, policy_fn)
            # new_obs = np.array(new_obs)
            # new_acts = np.array(new_acts)

            new_obs, _ = run_env2(net, args.envname, 20)
            new_obs = np.array(new_obs)
            new_acts = policy_fn(new_obs)
            new_dataset = data.TensorDataset(t.Tensor(new_obs), t.Tensor(new_acts).squeeze(1))
            datasets.append(new_dataset)
            all_dataset = data.ConcatDataset(datasets)
            dataloader = DataLoader(all_dataset, batch_size=200)
            print("len(all_dataset)", len(all_dataset))
            print("len(dataloader)", len(dataloader))
            
            epoch -= 1

        epoch = 1
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

        rollouts = 20
        _, returns = run_env2(net, args.envname, rollouts)

        log_file = os.path.join(PROFECT_ROOT, 'data', "Daggerlog.txt")

        import sys
        sys.stdout = open(log_file, 'a')

        print('returns for', args.envname)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()