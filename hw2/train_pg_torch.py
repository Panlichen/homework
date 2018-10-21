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
        self.n_layers = args.n_layers
        self.hidden_shape = args.hidden_shape
        self.lr = args.learning_rate
        self.ob_dims = env.observation_space.shape[0]
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.ac_dims = env.action_space.n if self.discrete else env.action_space.shape[0]
        
        # sample args
        self.render = args.render
        self.max_path_len = args.path_len if args.path_len > 0 else env.spec.max_episode_steps
        self.min_timesteps_per_batch = args.batch_shape

        # estimate args
        self.gamma = args.discount
        self.causality = args.causality
        self.need_normalize_adv = not(args.dont_normalize_advantange)

        self.policy_nn = PolicyNet(self.n_layers, self.ob_dims, self.ac_dims, self.hidden_shape, self.discrete)
        self.value_nn = build_mlp(self.ob_dims, 1, self.n_layers, self.hidden_shape)
        params = list(self.policy_nn.parameters()) + list(self.value_nn.parameters())
        self.optimizer = optim.Adam(params, lr=self.lr)

    
    
    def sample_action(self, ob_no):
        """
        这里体现出离散和连续的区别，对离散的action，每一维对应一个动作，这些动作互斥，计算每个动作的概率（和为1）
        而连续的action，每一维对应一个动作，所有动作不互斥，计算每个动作的浮点值
        """
        ts_ob_no = t.from_numpy(ob_no).float()
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
        return ts_sampled_ac.numpy()
        

    def sample_trajectory(self, render):
        ob = self.env.reset()
        obs, acs, rewards = [], [], []
        steps = 0 
        while True:
            if render:
                self.env.render()
                time.sleep(0.1)
            obs.append(ob)
            if DEBUG_POLICY:
                print('ob',ob)
            
            ac = self.sample_action(ob[None])
            ac = ac[0]
            # 虽然这里ac的类型是<class 'numpy.int64'>，但是acs的类型依旧是倔强的<class 'list'>
            acs.append(ac)
            ob, reward, done, _ = self.env.step(ac)
            rewards.append(reward)
            steps += 1
            # max_path_len: 控制每一次取样
            if done or steps > self.max_path_len:
                break
        
        path = {
            'ob': t.tensor(obs, dtype=t.float32),
            'ac': t.tensor(acs, dtype=t.float32),
            'reward': t.tensor(rewards, dtype=t.float32),
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
    
    def monte_carlo_q(self, ts_re_n):
        q_n = []
        if self.causality:
            for re in ts_re_n:
                path_len = len(re)
                gamma_power = t.pow(self.gamma, t.arange(path_len).float())
                q_path= [t.sum(re[t:] * gamma_power[:path_len - 1 - t]) for t in range(path_len)]
                q_n.extend(q_path)
        else:
            for re in ts_re_n:
                path_len = len(re)
                gamma_power = t.pow(self.gamma, t.arange(path_len).float())
                q_path = t.full((path_len, ), t.sum(re * gamma_power))
                """
                tensor之间做加减乘除，必须是相同类型~
                """
                q_n.extend(q_path)
        return t.tensor(q_n)
    
    def compute_advantage(self, ts_ob_no, ts_q_n):
        ts_v_n = self.value_nn(ts_ob_no).view(-1)
        raw_ts_v_n = ts_v_n
        # 网络的输出是(n, 1)，这里把它拉成(n, )的
        # print("compute_advantage nn out ts_v_n", ts_v_n)
        # 我们下边在训练value_nn的时候，target是normalize的q_n，所以可以预期下边两步操作对ts_v_n的影响不会很大。
        ts_v_n = (ts_v_n - t.mean(ts_v_n)) / (t.std(ts_v_n, unbiased=False) + 1e-7)
        ts_v_n = ts_v_n * t.std(ts_q_n, unbiased=False) + t.mean(ts_q_n)
        ts_adv_n = ts_q_n - ts_v_n
        if DEBUG_Q:
            print("^^^^^^^^^^^^^^^^^^^^^^^compute_advantage^^^^^^^^^^^^^^^^^^^^^^^")
            print('ts_ob_no', ts_ob_no)
            print('ts_q_n', ts_q_n)
            print('t.std(ts_q_n)', t.std(ts_q_n, unbiased=False))
            print('t.mean(ts_q_n)', t.mean(ts_q_n))
            print('raw_ts_v_n', raw_ts_v_n)
            print('ts_v_n', ts_v_n)
            print('t.std(ts_v_n)', t.std(ts_v_n, unbiased=False))
            print('t.mean(ts_v_n)', t.mean(ts_v_n))
            print('ts_adv_n', ts_adv_n)
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

        return ts_adv_n

    def estimate_return(self, ts_ob_no, ts_re_n):
        ts_q_n = self.monte_carlo_q(ts_re_n)
        ts_adv_n = self.compute_advantage(ts_ob_no, ts_q_n)
        if self.need_normalize_adv:
            ts_adv_n = (ts_adv_n - t.mean(ts_adv_n)) / (t.std(ts_adv_n, unbiased=False) + 1e-7)
        return ts_q_n, ts_adv_n
    
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

    def update_parameters(self, ts_ob_no, ts_ac_na, ts_adv_n, ts_q_n):
        ts_policy_out = self.policy_nn(ts_ob_no)
        ts_logprob_n = self.get_log_prob(ts_policy_out, ts_ac_na)
        
        self.optimizer.zero_grad()

        pseudo_policy_loss = -(ts_logprob_n * ts_adv_n).mean()
        pseudo_policy_loss.backward()

        ts_v_n = self.value_nn(ts_ob_no).view(-1)#!!!!!!!!!!!!!!!
        ts_target_n = (ts_q_n - t.mean(ts_q_n)) / (t.std(ts_q_n) + 1e-7)
        # value_loss_fn = nn.MSELoss()
        # value_loss = value_loss_fn(ts_v_n, ts_target_n)
        value_loss = nn.functional.mse_loss(ts_v_n, ts_target_n)
        value_loss.backward()
        if DEBUG_Q:
            print('################ update_parameters ###################')
            print('ts_policy_out', ts_policy_out)
            print('ts_logprob_n', ts_logprob_n)
            print('pseudo_policy_loss', pseudo_policy_loss)
            print('ts_v_n', ts_v_n)
            print('ts_target_n', ts_target_n)
            print('value_loss', value_loss)
            print('######################################################')

        self.optimizer.step()  
        return pseudo_policy_loss, value_loss     

def train_pg(args, env):
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
        # 这里reward不要连接起来，还是要保留各个路径的信息，但是最终的q_n应该是连接起来的，对应于每个ac有一个q值的估计。
        ts_re_n = [path['reward'] for path in paths]
        with t.no_grad():
            ts_q_n, ts_adv_n = agent.estimate_return(ts_ob_no, ts_re_n)
        pseudo_policy_loss, value_loss = agent.update_parameters(ts_ob_no, ts_ac_na, ts_adv_n, ts_q_n)
        pseudo_policy_losses.append(pseudo_policy_loss)
        value_losses.append(value_loss)
        returns = [path["reward"].sum() for path in paths]
        mean_return = t.mean(t.tensor(returns))
        if not DEBUG_POLICY and not DEBUG_Q:
            print(len(ts_re_n), len(returns))
            print('mean rewards: ', mean_return)
        # print('returns:', returns)

        mean_returns.append(mean_return)
        # ep_len = [len(path["reward"]) for path in paths]
        # print('mean ep_path_len', np.mean(ep_len))   
    vis = visdom.Visdom(env=u'test1')
    vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(pseudo_policy_losses), \
        win='pseudo_policy_losses', opts={'title': 'pseudo_policy_losses'})
    vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(value_losses), \
        win='value_losses', opts={'title': 'value_losses'})
    vis.line(X=t.tensor(range(args.n_iter)), Y=t.tensor(mean_returns), \
        win='mean_returns', opts={'title': 'mean_returns'})


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
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--causality', '-cau', action='store_true')
    parser.add_argument('--dont_normalize_advantange', '-dna', action='store_true')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--n_experiment', '-e', type=int ,default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_shape', '-hs', type=int, default=64)
    args=parser.parse_args()
    processes = []

    for e in range(args.n_experiment):
        # set up env
        env = gym.make(args.envname)
        seed = args.seed + 10 * e
        t.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
        print("Runnint with seed %d"%seed)
        
        train_pg(args, env)
        env.close()

        # def train_func():
        #     train_pg(args, env)
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


    # # deprecated, not as flexible as define a new nn.Module
    # def build_nn(self):
    #     """
    #     # Sequential的三种写法
    #     net1 = nn.Sequential()
    #     net1.add_module('conv', nn.Conv2d(3, 3, 3))
    #     net1.add_module('batchnorm', nn.BatchNorm2d(3))
    #     net1.add_module('activation_layer', nn.ReLU())

    #     net2 = nn.Sequential(
    #             nn.Conv2d(3, 3, 3),
    #             nn.BatchNorm2d(3),
    #             nn.ReLU()
    #             )

    #     from collections import OrderedDict
    #     net3= nn.Sequential(OrderedDict([
    #             ('conv1', nn.Conv2d(3, 3, 3)),
    #             ('bn1', nn.BatchNorm2d(3)),
    #             ('relu1', nn.ReLU())
    #             ]))
    #     print('net1:', net1)
    #     print('net2:', net2)
    #     print('net3:', net3)
    #     """
    #     self.policy_nn = nn.Sequential()
    #     self.policy_nn.add_module('line0', nn.Linear(self.ob_dims, self.hidden_shape))
    #     self.policy_nn.add_module('relu0', nn.ReLU())
    #     for i in range(self.n_layers - 1):
    #         self.policy_nn.add_module('line' + str(i + 1), nn.Linear(self.hidden_shape, self.hidden_shape))
    #         self.policy_nn.add_module('relu' + str(i + 1), nn.ReLU())
    #     self.policy_nn.add_module('line' + str(self.n_layers), nn.Linear(self.hidden_shape, self.ac_dims))
    #     if self.discrete:
    #         self.get_prob = nn.Softmax(dim=0)

    #     self.value_nn = nn.Sequential()
    #     self.value_nn.add_module('line0', nn.Linear(self.ob_dims, self.hidden_shape))
    #     self.value_nn.add_module('relu0', nn.ReLU())
    #     for i in range(self.n_layers - 1):
    #         self.value_nn.add_module('line' + str(i + 1), nn.Linear(self.hidden_shape, self.hidden_shape))
    #         self.value_nn.add_module('relu' + str(i + 1), nn.ReLU())
    #     self.value_nn.add_module('line' + str(self.n_layers), nn.Linear(self.hidden_shape, 1))