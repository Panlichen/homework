"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
Adapted for pytorch by Lichen Pan
"""
import numpy as np
import torch as t
import gym
import os
import time
import inspect

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--exp_name', type=str, default='temp')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--path_len', '-pl', type=float, default=-1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--causality', '-cau', action='store_true')
    parser.add_argument('--dont_normalize_advantange', '-dna', action='store_true')
    parser.add_argument('--seed', type=float, default=1)
    parser.add_argument('--n_experiment', '-e', type=int ,default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_size', '-hs', type=int, default=64)
    args=parser.parse_args()
    
    max_path_length = args.path_len if args.path_len > 0 else None
    

if __name__ == "__main__":
    main()