import argparse

import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import visdom

from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.car_racing.controller import key_phrase
from competitive_rl.car_racing.register import register_competitive_envs

register_competitive_envs()

class DrawLine():

    def __init__(self, env, title, xlabel=None, ylabel=None):
        self.vis = visdom.Visdom()
        self.update_flag = False
        self.env = env
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title

    def __call__(
            self,
            xdata,
            ydata,
    ):
        if not self.update_flag:
            self.win = self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                opts=dict(
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                    title=self.title,
                ),
                env=self.env,
            )
            self.update_flag = True
        else:
            self.vis.line(
                X=np.array([xdata]),
                Y=np.array([ydata]),
                win=self.win,
                env=self.env,
                update='append',
            )

parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=8, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--vis', action='store_true', help='use visdom')
parser.add_argument(
    '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

transition = np.dtype([('s', np.float64, (args.img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                       ('r', np.float64), ('s_', np.float64, (args.img_stack, 96, 96))])


class Env():
    """
    Test environment wrapper for CarRacing
    """

    def __init__(self,num_player=1):
        #self.env = gym.make('cCarRacing-v0')
        self.num_palyer = num_player
        self.env = CarRacing(num_player=num_player)
        self.env.seed(args.seed)
        #self.reward_threshold = self.env.spec.reward_threshold

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.dies = [False] * self.num_palyer
        img_rgbs = self.env.reset()
        img_grays = [self.rgb2gray(img_rgbs[i]) for i in range(self.num_palyer)]
        self.stacks = [[img_grays[i]] * args.img_stack for i in range(self.num_palyer)]
        return [np.array([img_grays[i]] * args.img_stack) for i in range(self.num_palyer)]

    def step(self, actions):
        total_rewards = [0] * self.num_palyer
        img_rgb = []
        for i in range(args.action_repeat):
            img_rgbs, rewards, dies, _ = self.env.step(actions)
            dones = [False] * self.num_palyer
            img_rgb = img_rgbs
            for i in range(self.num_palyer):
                # don't penalize "die state"
                if dies[i]:
                    rewards[i] += 100
                # green penalty
                if np.mean(img_rgbs[i][:, :, 1]) > 185.0:
                    rewards[i] -= 0.05
                total_rewards[i] += rewards[i]
                # if no reward recently, end the episode
                done = True if self.av_r(rewards[i]) <= -0.1 else False
                dones[i] = done
                if done or dies[i]:
                #if dies[i]:
                    break
        img_grays = [self.rgb2gray(img_rgb[i]) for i in range(self.num_palyer)]
        for i in range(self.num_palyer):
            self.stacks[i].pop(0)
            self.stacks[i].append(img_grays[i])
            assert len(self.stacks[i]) == args.img_stack
        #return np.array(self.stack), total_reward, done, die
        return [np.array(self.stacks[i]) for i in range(self.num_palyer)], total_rewards, dones, dies
        #return [np.array(self.stacks[i]) for i in range(self.num_palyer)], total_rewards, [False], dies

    def render(self, *arg):
        self.env.render(*arg)

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if norm:
            # normalize
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class Agent_to_Train():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss
    ppo_epoch = 10
    buffer_capacity, batch_size = 2000, 128

    def __init__(self):
        self.training_step = 0
        self.net = Net().double().to(device)
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self, save_path='param/car0.0.pkl'):
        torch.save(self.net.state_dict(), save_path)

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + args.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                loss = action_loss + 2. * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def load_param(self,load_path='param/car0.2.pkl'):
        self.net.load_state_dict(torch.load(load_path))

class Trained_Agent():
    """
    Agent for testing
    """

    def __init__(self):
        self.net = Net().float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self,load_path='param/car0.2.pkl'):
        self.net.load_state_dict(torch.load(load_path))

if __name__ == "__main__":
    num_player = 2
    agent_to_train_1 = Agent_to_Train()
    #agent_to_train_1.load_param('param/car0.3 - 副本 (5).pkl')
    agent_to_train = Agent_to_Train()
    #agent_to_train.load_param('param/car0.3 - 副本 (5).pkl')

    env = Env(num_player=num_player)
    #if args.vis:
    draw_reward = DrawLine(env="car", title="PPO", xlabel="Episode", ylabel="Moving averaged episode reward")

    training_records = []
    running_score = 0
    states = env.reset()
    a = [[0, 0, 0] * num_player]
    for i_ep in range(100000):
        score = 0
        states = env.reset()

        for t in range(1000):
            env.env.manage_input(key_phrase(a))
            action1, a_logp_1 = agent_to_train.select_action(states[0])
            action2,  a_logp_2= agent_to_train_1.select_action(states[1])
            actions = [action1 * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]), action2 * np.array([2., 1., 1.]) + np.array([-1., 0., 0.])]
            states_, rewards, done, die = env.step(actions)
            #if args.render:
            if env.env.isrender:
                env.render()
            if agent_to_train.store((states[0], action1, a_logp_1, rewards[0], states_[0])):
                print('updating agent_to_train')
                agent_to_train.update()
            if agent_to_train_1.store((states[1], action2, a_logp_2, rewards[1], states_[1])):
                print('updating agent_to_train_1')
                agent_to_train_1.update()
            score += rewards[0]
            states = states_
            if any(die) or any(done):
                break
        running_score = running_score * 0.99 + score * 0.01

        if i_ep % args.log_interval == 0:
            #if args.vis:
            draw_reward(xdata=i_ep, ydata=running_score)
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))
            agent_to_train.save_param('param/car0.3.pkl')
            #agent_to_train_1.save_param('param/car0.3.pkl')
        if running_score > 900:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
