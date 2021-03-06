#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:]) # 从加载的policy中读ob对应的action
                observations.append(obs) # 这行和下一行只是为了记录
                actions.append(action)
                # print("obs[None,:]", obs[None,:].shape)
                # print("action", action.shape)
                obs, r, done, _ = env.step(action) # 在起来的环境中运行action，会得到新的ob
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: 
                    print("%i/%i"%(steps, max_steps))
                    # print(actions)
                    # print(obs)
                if steps >= max_steps:
                    break
            returns.append(totalr)
        PROFECT_ROOT = os.path.dirname(os.path.realpath(__file__))
        log_file = os.path.join(PROFECT_ROOT, 'data', "expertlog.txt")
        import sys
        sys.stdout = open(log_file, 'a')
        
        print('returns for', args.envname)
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # get train data

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
        save_dir = os.path.join(PROJECT_ROOT, "data/")
        out_file = os.path.join(save_dir, args.envname+'.train')
        np.savez(out_file, expert_data['observations'], expert_data['actions'])




        # with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        #     pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
