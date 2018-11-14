"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Adapted for pytorch by Lichen Pan
"""
import numpy as np
import torch as t
from torch import nn
from torch import  optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.multiprocessing import Process
import gym
import os
import time
import inspect
import visdom

DEBUG_POLICY = False
DEBUG_Q = False
t.set_printoptions(precision=6, threshold=1e9)
np.set_printoptions(precision=6)
def build_mlp(input_shape, output_shape, n_layers, hidden_shape, activation=nn.Tanh):
    """
    mlp for multilayer perceptron
    激活函数很重要的！！
    不过不知道为啥nn.Linear(input_shape, hidden_shape, activation())不报错。。
    """
    layers = []
    for _ in range(n_layers):
        layers.append(nn.Linear(input_shape, hidden_shape))
        layers.append(activation())
        input_shape = hidden_shape
    layers.append(nn.Linear(hidden_shape, output_shape))
    return nn.Sequential(*layers).apply(weights_init)

def weights_init(m):
    if hasattr(m, 'weight'):
        nn.init.xavier_uniform_(m.weight)

class PolicyNet(nn.Module):
    def __init__(self, n_layers, ob_dims, ac_dims, hidden_shape, discrete):
        super(PolicyNet, self).__init__()
        self.discrete = discrete
        self.mlp = build_mlp(ob_dims, ac_dims, n_layers, hidden_shape)
        if not self.discrete:
            self.logstd = nn.Parameter(t.randn((ac_dims, )))
    def forward(self, ts_ob_no):
        if self.discrete:
            ts_logits_na = self.mlp(ts_ob_no)
            return ts_logits_na
        else:
            ts_mean_na = self.mlp(ts_ob_no)
            ts_logstd_na = self.logstd
            return (ts_mean_na, ts_logstd_na)
        

class Agent(object):
    def __init__(self, args, env):
        super(Agent, self).__init__()
        # nn args
        self.env = env
        self.critic_n_layers = args.critic_n_layers
        self.actor_n_layers = args.actor_n_layers
        self.critic_learning_rate = args.critic_learning_rate
        self.actor_learning_rate = args.actor_learning_rate
        self.num_target_updates = args.num_target_updates
        self.num_grad_steps_per_target_update = args.num_grad_steps_per_target_update
        self.hidden_shape = args.hidden_shape
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.gpu = args.gpu


        # print(env.observation_space.shape)
        # print(env.observation_space.shape[0])
        # print(env.action_space)
        # print(self.discrete)
        # print(env.action_space.n if self.discrete else env.action_space.shape)
        # print(env.action_space.n if self.discrete else env.action_space.shape[0])
        # os._exit(1)


        if args.envname == 'Copy-v0':
            self.ob_dims = env.observation_space.shape
            self.ac_dims = env.action_space.n if self.discrete else env.action_space.shape
        elif args.envname == 'Zaxxon-v0' or args.envname == 'CarRacing-v0':
            self.ob_dims = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
            self.ac_dims = env.action_space.n if self.discrete else env.action_space.shape[0]
        else:
            self.ob_dims = env.observation_space.shape[0]
            self.ac_dims = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        # sample args
        self.render = args.render
        self.max_path_len = args.path_len if args.path_len > 0 else env.spec.max_episode_steps
        self.min_timesteps_per_batch = args.batch_shape

        # estimate args
        self.gamma = args.discount
        self.causality = args.causality
        self.need_normalize_adv = not(args.dont_normalize_advantange)

        self.policy_nn = PolicyNet(self.actor_n_layers, self.ob_dims, self.ac_dims, self.hidden_shape, self.discrete)
        self.value_nn = build_mlp(self.ob_dims, 1, self.critic_n_layers, self.hidden_shape)
        if self.gpu:
            self.policy_nn = self.policy_nn.cuda()
            self.value_nn = self.value_nn.cuda()
        
        self.actor_optimizer = optim.Adam(self.policy_nn.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.value_nn.parameters(), lr=self.critic_learning_rate)

    
    
    def sample_action(self, ob_no):
        """
        这里体现出离散和连续的区别，对离散的action，每一维对应一个动作，这些动作互斥，计算每个动作的概率（和为1）
        而连续的action，每一维对应一个动作，所有动作不互斥，计算每个动作的浮点值
        """
        ts_ob_no = t.from_numpy(ob_no).float()
        if self.gpu:
            ts_ob_no = ts_ob_no.cuda()
        if self.discrete:
            ts_logits_na = self.policy_nn(ts_ob_no)
            ts_prob_na = nn.functional.log_softmax(ts_logits_na, dim=-1).exp()
            ts_sampled_ac = t.multinomial(ts_prob_na, 1).view(-1)
            # print("ts_sampled_ac.squeeze(-1)", ts_sampled_ac.squeeze(-1))
            # print("ts_sampled_ac.view(-1)", ts_sampled_ac.view(-1))
            # view(*args) → Tensor; args (torch.Size or int...) – the desired size
            # view是直接指定输出tensor的维度，squeeze是压缩所有为1的维度，这里直接调用squeeze的话，离散情况下会压成一个标量
            if DEBUG_POLICY:
                print("ts_logits_na", ts_logits_na)
                print("ts_logits_na.shape", ts_logits_na.shape)
                print("ts_prob_na", ts_prob_na)
                print("ts_sampled_ac", ts_sampled_ac)
        else:
            ts_mean_na, ts_logstd_na = self.policy_nn(ts_ob_no)
            ts_sampled_ac = t.normal(ts_mean_na, ts_logstd_na)
        if self.gpu:
            ts_sampled_ac = ts_sampled_ac.cpu()
        return ts_sampled_ac.numpy()
        

    def sample_trajectory(self, render):
        ob = self.env.reset()
        obs, acs, rewards, next_obs, terminals = [], [], [], [], []
        steps = 0 
        while True:
            if render:
                self.env.render()
                time.sleep(0.1)
            ob = ob.flatten()
            obs.append(ob)
            # print(type(ob))
            # print(t.tensor(ob).shape)
            # os._exit(1)
            if DEBUG_POLICY:
                print('ob',ob)
            
            ac = self.sample_action(ob[None])
            ac = ac[0]
            # 虽然这里ac的类型是<class 'numpy.int64'>，但是acs的类型依旧是倔强的<class 'list'>
            acs.append(ac)
            ob, reward, done, _ = self.env.step(ac)


            # add the observation after taking a step to next_obs
            next_obs.append(ob)

            rewards.append(reward)
            steps += 1
            # max_path_len: 控制每一次取样
            # If the episode ended, the corresponding terminal value is 1
            # otherwise, it is 0
            if done or steps > self.max_path_len:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {
            'ob': t.tensor(obs, dtype=t.float32),
            'ac': t.tensor(acs, dtype=t.float32),
            'reward': t.tensor(rewards, dtype=t.float32),
            'next_ob': t.tensor(next_obs, dtype=t.float32),
            'terminal': t.tensor(terminals, dtype=t.float32)
        }
        return path, steps

        
    def sample_trajectories(self, itr):
        steps_this_batch = 0
        paths = []
        while True:
            render = (len(paths) == 0 and itr % 10 == 0 and self.render)
            path, steps = self.sample_trajectory(render)
            paths.append(path)
            steps_this_batch += steps
            # min_timesteps_per_batch：控制每个batch的总长度
            if steps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, steps_this_batch

    def update_critic(self, ts_ob_no, ts_next_ob_no, ts_re_n, ts_terminal_n):
        for _ in range(self.num_target_updates):
            with t.no_grad():
                ts_target_n = ts_re_n + self.gamma * self.value_nn(ts_next_ob_no).view(-1) * (1 - ts_terminal_n)
            for _ in range(self.num_grad_steps_per_target_update):
                ts_value_n = self.value_nn(ts_ob_no).view(-1)

                self.critic_optimizer.zero_grad()
                value_loss = nn.functional.mse_loss(ts_value_n, ts_target_n)
                value_loss.backward()
                self.critic_optimizer.step()
    
    def estimate_advantage(self, ts_ob_no, ts_next_ob_no, ts_re_n, ts_terminal_n):
        with t.no_grad():
            ts_q_n = ts_re_n + self.gamma * self.value_nn(ts_next_ob_no).view(-1) * (1 - ts_terminal_n)
            ts_v_n = self.value_nn(ts_ob_no).view(-1)
        ts_adv_n = ts_q_n - ts_v_n
        return ts_adv_n
    
    def get_log_prob(self, ts_policy_out, ts_ac_na):
        if self.discrete:
            ts_logits_na = ts_policy_out
            # 这里的ts_ac_na的实际维度应该是n，没有a。
            ts_logprob_n = t.distributions.Categorical(logits=ts_logits_na).log_prob(ts_ac_na)
        else:
            ts_mean_na, ts_logstd_na = ts_policy_out
            # 对连续动作，之所以我们拟合logstd，是因为我们的NN或者这里的随机初始化出来的可能是个负数，不能直接用作标准差。
            # 这里的ts_ac_na的实际维度应该是na。
            # 连续情况下，我们求的是各个分动作各自的概率，而没有求联合概率，这样会很方便，在计算梯度时候加一下和，不影响梯度的更新。
            ts_logprob_n = t.distributions.Normal(loc=ts_mean_na, scale=ts_logstd_na.exp()).log_prob(ts_ac_na).sum(-1)
            # tensor.sum()和tensor .sum()效果是一样的，厉害了。。
        return ts_logprob_n

    def update_actor(self, ts_ob_no, ts_ac_na, ts_adv_n):
        ts_policy_out = self.policy_nn(ts_ob_no)
        ts_logprob_n = self.get_log_prob(ts_policy_out, ts_ac_na)
        self.actor_optimizer.zero_grad()
        pseudo_policy_loss = -(ts_logprob_n * ts_adv_n).mean()
        pseudo_policy_loss.backward()
        self.actor_optimizer.step()


def train_ac(args, env):
    agent = Agent(args, env)

    # input()
    mean_returns = []
    pseudo_policy_losses = []
    value_losses = []
    for itr in range(args.n_iter):
        print("********** Iteration %i ************"%itr)
        # input()
        if DEBUG_POLICY or DEBUG_Q:
            print('net', agent.policy_nn)
            for name, parameter in agent.policy_nn.named_parameters():
                print(name, parameter)
            print('value_net', agent.value_nn)
            for name, parameter in agent.value_nn.named_parameters():
                print(name, parameter)
        with t.no_grad():
            paths, steps_this_batch = agent.sample_trajectories(itr)
        # [...]的作用是生成一个大的list（之所以不用小括号括起来，是因为那将构成一个tuple，无法编辑），然后对这个大list进行concatenate操作
        # （即每个元素“抽去最外层，合并成一个”，这个说法不科学，因为有axis=参数，用到时候再具体实验理解）。
        ts_ob_no = t.cat([path['ob'] for path in paths])
        # 对于离散的情况，取样的ac的真实维度应该是ac_n，即index值，这样也不影响去计算CrossEntropy loss
        ts_ac_na = t.cat([path['ac'] for path in paths])
        ts_re_n = t.cat([path['reward'] for path in paths])
        ts_next_ob_no = t.cat([path['next_ob'] for path in paths])
        ts_terminal_n = t.cat([path['terminal'] for path in paths])
        if agent.gpu:
            ts_ob_no, ts_ac_na, ts_re_n, ts_next_ob_no, ts_terminal_n = map(lambda x: x.cuda(), \
                [ts_ob_no, ts_ac_na, ts_re_n, ts_next_ob_no, ts_terminal_n])
        
        agent.update_critic(ts_ob_no, ts_next_ob_no, ts_re_n, ts_terminal_n)
        ts_adv_n = agent.estimate_advantage(ts_ob_no, ts_next_ob_no, ts_re_n, ts_terminal_n)
        agent.update_actor(ts_ob_no, ts_ac_na, ts_adv_n)

        
        
        #
        #  with t.no_grad():
        #     ts_q_n, ts_adv_n = agent.estimate_return(ts_ob_no, re_n)
        # pseudo_policy_loss, value_loss = agent.update_parameters(ts_ob_no, ts_ac_na, ts_adv_n, ts_q_n)
        # pseudo_policy_losses.append(pseudo_policy_loss)
        # value_losses.append(value_loss)
        returns = [path["reward"].sum() for path in paths]
        mean_return = t.mean(t.tensor(returns))
        if not DEBUG_POLICY and not DEBUG_Q:
            print(len(returns))
            print('mean rewards: ', mean_return)
        # print('returns:', returns)

        mean_returns.append(mean_return)
        # ep_len = [len(path["reward"]) for path in paths]
        # print('mean ep_path_len', np.mean(ep_len))   
    # vis = visdom.Visdom(env=u'test1')
    # vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(pseudo_policy_losses), \
    #     win='pseudo_policy_losses', opts={'title': 'pseudo_policy_losses'})
    # vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(value_losses), \
    #     win='value_losses', opts={'title': 'value_losses'})
    # vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(mean_returns), \
    #     win='mean_returns', opts={'title': 'mean_returns'})


def main():
    
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--exp_name', type=str, default='temp')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_shape', '-b', type=int, default=1000)
    parser.add_argument('--path_len', '-pl', type=float, default=-1.0)
    parser.add_argument('--actor_learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--critic_learning_rate', '-clr', type=float)
    parser.add_argument('--causality', '-cau', action='store_true')
    parser.add_argument('--dont_normalize_advantange', '-dna', action='store_true')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--n_experiment', '-e', type=int ,default=1)
    parser.add_argument('--actor_n_layers', '-l', type=int, default=2)
    parser.add_argument('--critic_n_layers', '-cl', type=int)
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
    parser.add_argument('--hidden_shape', '-hs', type=int, default=64)
    parser.add_argument('--gpu', action='store_true')
    args=parser.parse_args()

    if not args.critic_learning_rate:
        args.critic_learning_rate = args.actor_learning_rate
    if not args.critic_n_layers:
        args.critic_n_layers = args.actor_n_layers

    for e in range(args.n_experiment):
        # set up env
        env = gym.make(args.envname)
        seed = args.seed + 10 * e
        t.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
        print("Runnint with seed %d"%seed)
        
        train_ac(args, env)
        env.close()

        # def train_func():
        #     train_ac(args, env)
        # p = Process(target=train_func, args=tuple())
        # p.start()
        # processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        # p.join()

    # for p in processes:
    #     p.join()
    

if __name__ == "__main__":
    main()
