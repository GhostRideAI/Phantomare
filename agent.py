# Imports
from gymnasium.core import Env
import torch
from torch import Tensor, nn 
import torch.distributions as td
import gymnasium as gym
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from tensordict import TensorDict
from utils import Utils

#####################################################################
#   Environments 
#####################################################################

class Environment:

    def __init__(self, gym_name: str, record: bool = False) -> None:
        self.gym = self._create_gym(gym_name, record)

    def reset(self) -> tuple[Tensor]:
        exp = self.gym.reset()
        return exp
    
    def step(self, action: Tensor) -> tuple[Tensor]:
        obs, reward, term, trunc, _ = self.gym.step(action)
        term = torch.tensor([[term]], dtype=torch.bool)
        trunc = torch.tensor([[trunc]], dtype=torch.bool)
        return obs, reward, term, trunc
 
    def _create_gym(self, name: str, record: bool) -> gym.Env:
        if record:
            env = gym.make(name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, './', 
                disable_logger=True)
        else:
            env = gym.make(name)
        env = self.ActionProcessor(env)
        env = self.ObservationProcessor(env)
        env = self.RewardProcessor(env)
        return env

    class RewardProcessor(gym.RewardWrapper):

        def __init__(self, env: Env):
            super().__init__(env)

        def reward(self, reward: float) -> Tensor:
            return torch.tensor([[reward]], dtype=torch.float32)

    class ObservationProcessor(gym.ObservationWrapper):

        def __init__(self, env: Env) -> None:
            super().__init__(env)

        def observation(self, observations: dict[str, NDArray] | NDArray) -> Tensor:
            processed = TensorDict()
            if not isinstance(observations, dict):
                if observations.ndim > 1:
                    observations = {'image': observations}
                else:
                    observations = {'observation': observations}
            for name, obs in observations.items():
                # 2-D observation
                if obs.ndim > 1:
                    if obs.shape[-1] == 1:
                        obs = Image.fromarray(obs[...,0])
                    else:
                        obs = Image.fromarray(obs)
                    obs = Utils.img2ten(obs).unsqueeze(0)
                # 1-D observation
                else:
                    obs = torch.tensor([obs], dtype=torch.float32)
                processed[name] = obs
            processed.auto_batch_size_(1)
            return processed 

    class ActionProcessor(gym.ActionWrapper):

        def __init__(self, env: Env):
            super().__init__(env)

        def action(self, action: Tensor) -> NDArray:
            action = action.squeeze(0).cpu().numpy()
            l = self.action_space.low
            h = self.action_space.high
            action = (h - l) * (action + 1)/2 + l
            return action.astype(np.float64)
        
#####################################################################
#   Agent
#####################################################################

class Agent:

    def __init__(self, env: Environment, policy: str | tuple[nn.Module],
                 device: str = 'cpu', action_repeat: int = 1, 
                 expl_noise: float = 0.0) -> None:
        self.env = env
        self.action_repeat = action_repeat
        self.expl_noise = expl_noise
        self.device = torch.device(device)
        self.current_return = 0.0
        self.update_policy(policy)

    def reset(self) -> Tensor:
        self.current_return = 0.0
        if self._policy_type == 'nn':
            self.h_t = self._world_model._get_inital_recurrent_state(1)
        return self.env.reset()
    
    def act(self, obs: tuple, exploit: bool=False) -> tuple[Tensor]:
            action = self.policy(obs, exploit)
            total_reward = []
            for _ in range(self.action_repeat):
                obs, reward, term, trunc = self.env.step(action)
                total_reward += [reward]
                if max(term, trunc): break
            reward = sum(total_reward)
            self.current_return += reward.item()
            return obs, action, reward, term, trunc

    def update_policy(self, policy: str | tuple[nn.Module]) -> None:
        if policy == 'random':
            self._policy_type = policy
            self.policy = self._random_policy
        else:
            self._policy_type = 'nn'
            self._world_model, self._actor = [
                m.to(self.device).eval() for m in policy]
            self.policy = self._nn_policy

    def _random_policy(self, *args, **kwargs) -> Tensor:
        return torch.zeros(1,*self.env.gym.action_space.shape).uniform_(-1, 1)

    @torch.no_grad()
    def _nn_policy(self, observation: TensorDict, exploit: bool) -> Tensor:
        if not hasattr(self, 'h_t'):
            self.h_t = self._world_model._get_inital_recurrent_state(1)
        observation = observation.to(device=self.device)
        for name, obs in observation.items():
            if obs.dtype == torch.uint8:
                observation[name] = obs.to(dtype=torch.float32) / 255.0
        state, posterior = self._world_model.encode_state(observation, self.h_t)
        action_dist = self._actor(state)
        if exploit: action = action_dist.mode
        else: 
            action = action_dist.rsample()
            if self.expl_noise:
                action = td.Normal(action, self.expl_noise).rsample().clamp(-1,1)
        za = torch.cat((posterior, action), -1)
        self.h_t = self._world_model.sequence_net(za, self.h_t)
        return action

