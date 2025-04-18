import os
import torch
import torch.nn as nn
import random
import pickle
from pathlib import Path

from blendrl.nsfr.nsfr.utils.common import load_module
from torch.distributions import Categorical
from ..env import NudgeBaseEnv


class ActorCritic(nn.Module):
    def __init__(self, env: NudgeBaseEnv, rng=None, device=None):
        super(ActorCritic, self).__init__()

        self.device = device
        self.rng = random.Random() if rng is None else rng
        self.env = env

        mlp_module_path = f"in/envs/{self.env.name}/mlp.py"
        module = load_module(mlp_module_path)
        self.actor = module.MLP(device=device, has_softmax=True)
        self.critic = module.MLP(device=device, has_softmax=False, out_size=1)

        self.n_actions = self.env.n_actions()
        self.uniform = Categorical(
            torch.tensor([1.0 / self.n_actions for _ in range(3)], device=device))

    def forward(self):
        raise NotImplementedError

    def act(self, state, epsilon=0.0):

        action_probs = self.actor(state)

        # e-greedy
        if self.rng.random() < epsilon:
            # random action with epsilon probability
            dist = self.uniform
        else:
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class NeuralPPO:
    def __init__(self, env: NudgeBaseEnv, lr_actor, lr_critic, optimizer, gamma,
                 epochs, eps_clip, device=None):

        self.device = device
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(env, device=device)
        self.optimizer = optimizer([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(env, device=device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, epsilon=0.0):
        # select random action with epsilon probability and policy probability with 1-epsilon
        with torch.no_grad():
            # state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state, epsilon=epsilon)

        self.buffer.states.append(state)
        action = torch.squeeze(action)
        self.buffer.actions.append(action)
        action_logprob = torch.squeeze(action_logprob)
        self.buffer.logprobs.append(action_logprob)

        return action

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # training does not converge if the entropy term is added ...
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards)  # - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            # wandb.log({"loss": loss})

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path, directory: Path, step_list, reward_list):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        with open(directory / "data.pkl", "wb") as f:
            pickle.dump(step_list, f)
            pickle.dump(reward_list, f)

    def load(self, directory: Path):
        # only for recover form crash
        model_name = input('Enter file name: ')
        model_file = os.path.join(directory, model_name)
        self.policy_old.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        with open(directory / "data.pkl", "rb") as f:
            step_list = pickle.load(f)
            reward_list = pickle.load(f)
        return step_list, reward_list

    def get_weights(self):
        return self.policy.actor.get_params()


class NeuralPlayer:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model
        self.device = torch.device('cuda:0')

    def act(self, state):
        logic_state, neural_state = state
        prediction = self.model(neural_state)
        action = torch.argmax(prediction).cpu().item() + 1
        return action


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
