# Code is heavily inspired by Tianshou's code. Please check out at https://github.com/thu-ml/tianshou
import argparse
import os
import pprint

import gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import TRPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Vehicle-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--step-per-epoch', type=int, default=5000)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument(
        '--repeat-per-collect', type=int, default=2 
    )  # theoretically it should be 1
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[128,128,128,128])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # trpo special
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--optim-critic-iters', type=int, default=5)
    parser.add_argument('--max-kl', type=float, default=0.005)
    parser.add_argument('--backtrack-coeff', type=float, default=0.8)
    parser.add_argument('--max-backtracks', type=int, default=10)

    args = parser.parse_known_args()[0]
    return args


def test_trpo(args=get_args()):
    env = gym.make(args.task)
    if args.task == 'Pendulum-v0':
        env.spec.reward_threshold = -250
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)
    test_envs = DummyVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device
    )
    actor = ActorProb(
        net,
        args.action_shape,
        max_action=args.max_action,
        unbounded=True,
        device=args.device
    ).to(args.device)
    critic = Critic(
        Net(
            args.state_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
            activation=nn.Tanh
        ),
        device=args.device
    ).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = TRPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
        optim_critic_iters=args.optim_critic_iters,
        max_kl=args.max_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'trpo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )


    log_path = os.path.join(args.logdir, args.task, "trpo")
    if False:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, "policy.pth")
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint)
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")






    if __name__ == '__main__':
        pprint.pprint(result)

        env = gym.make(args.task)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_trpo()