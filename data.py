# Imports
from typing import Any
import signal
import time
from typing import Iterator
import multiprocessing as mp
from multiprocessing.synchronize import Lock, Event
from multiprocessing.sharedctypes import Synchronized
from tqdm import tqdm
import torch
from torch import nn 
from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as ptl
from tensordict import TensorDict
from torchrl.data import ReplayBuffer
from agent import Agent, Environment
from utils import Config
    
#####################################################################
#   Data-Collectors
#####################################################################

class DataCollector:

    def __init__(self, replay_buffer: ReplayBuffer, 
                 policy: tuple[nn.Module] | str,
                 timesteps_per_sample: int,
                 agent_update_freq: int,
                 environment_args: Config,
                 agent_args: Config, rb_lock: Lock) -> None:
        self.replay_buffer = replay_buffer
        self.timesteps_per_sample = timesteps_per_sample
        self.agent_update_freq = agent_update_freq
        self.agent_args = agent_args
        self._return_buffer = []
        self.total_steps = 0
        self.rb_lock = rb_lock
        self._env_args = environment_args
        self._agent_args = agent_args
        self.env = Environment(**environment_args.to_dict())
        self.agent = Agent(self.env, policy, **agent_args.to_dict())
        self.action_repeat = self.agent.action_repeat
        self._current_observation, _ = self.agent.reset()
        self._current_experience = {'observations': [], 'actions': [],
                                    'rewards': [], 'continues': []}
        self._prev_steps = 0
        self._prev_time = time.time()
    
    def update_policy(self, policy: tuple[nn.Module] | str) -> None:
        self.agent.update_policy(policy)

    @property
    def average_return(self) -> float | None:
        if len(self._return_buffer) > 0:
            avg_return = sum(self._return_buffer)/len(self._return_buffer)
            self._return_buffer = []
            return avg_return
        else:
            return torch.nan
        
    @property
    def fps(self) -> float:
        prev_steps = self._prev_steps
        current_steps = self._prev_steps = self.total_steps
        prev_time = self._prev_time
        current_time = self._prev_time = time.time()
        fps = ((current_steps - prev_steps) / 
               (current_time - prev_time))
        return fps

    def step(self) -> None:
        # Step Agent
        obs = self._current_observation
        new_obs, action, reward, term, trunc = self.agent.act(obs, exploit=False)
        self.total_steps += 1
        # Determine if episode is done
        done = max(term, trunc)
        # Add step results to current experience
        self._current_experience['observations'] += [obs.cpu()]
        self._current_experience['actions'] += [action.cpu()]
        self._current_experience['rewards'] += [reward.cpu()]
        self._current_experience['continues'] += [~done.cpu()]
        # If episode is done...
        if done:
            # Reset the agent
            self._current_observation, _ = self.agent.reset()
        else: 
            # Else update the current observation
            #   with the new observation
            self._current_observation = new_obs
        # If the current experience is at the max number of timesteps for a sample...
        if len(self._current_experience['actions']) == self.timesteps_per_sample:
            # Convert experience into a TensorDict
            experience = TensorDict({k: torch.cat(v) for k,v in
                                     self._current_experience.items()})
            experience.auto_batch_size_(1)
            # Aquire the replay buffer lock and add the experience to it
            with self.rb_lock:
                self.replay_buffer.add(experience)
            # Initialize the new experience with all
            #   previous experience minus the 1st timestep
            for k,v in self._current_experience.items():
                self._current_experience[k] = v[1:]

    def evaluate_agent(self, n_episodes: int) -> None:
        # Create new environment for evaluation 
        env = Environment(**self._env_args.to_dict())
        # Get policy type from current agent
        policy_type = self.agent._policy_type
        # If policy type is random...
        if policy_type == 'random':
            # Set policy to random
            policy = 'random'
        else:
            # Else we get the neural networks from the current agent
            policy = (self.agent._world_model, self.agent._actor) 
        # Construct the new agent to be used for evaluation
        agent = Agent(env, policy, **self._agent_args.to_dict())
        # For N episodes, evaluate;
        for _ in range(n_episodes):
            obs, _ = agent.reset()
            done = False
            while not done:
                new_obs, _, _, term, trunc = agent.act(obs, exploit=True)
                obs = new_obs
                done = max(term, trunc)
            # Add agent's return to return buffer
            self._return_buffer += [agent.current_return]

class ConcurrentDataCollector:

    def __init__(self, replay_buffer: ReplayBuffer, 
                 policy: tuple[nn.Module] | str,
                 timesteps_per_sample: int,
                 agent_update_freq: int,
                 environment_args: Config,
                 agent_args: Config,
                 n_workers: int, rb_lock: Lock,
                 decoupled: bool=True) -> None:
        self._total_steps = mp.Value('I', 0)
        self.rb_lock = rb_lock
        self.decoupled = decoupled
        # BUG: workaround for TorchRL replay buffer 
        self._init_replay_buffer(
            replay_buffer, policy, timesteps_per_sample,
            agent_update_freq, environment_args, agent_args, rb_lock)
        self.agent_update_freq = agent_update_freq
        # Start process for data-collection
        self._replay_buffer = replay_buffer
        self._policy = policy
        self._timesteps_per_sample = timesteps_per_sample
        self._agent_update_freq = agent_update_freq
        self._environment_args = environment_args
        self._agent_args = agent_args
        self.n_workers = n_workers
        self.processes = []
        self.is_running = False

    def start(self) -> None:
        self.processes = [mp.Process(target=self._collect, args=(
            self._replay_buffer, self._policy, self._timesteps_per_sample,
            self._agent_update_freq, self._environment_args, self._agent_args,
            self._total_steps, self.rb_lock)) for _ in range(self.n_workers)]
        # Launch data-collection process
        [p.start() for p in self.processes]
        self.is_running = True
        self._prev_steps = 0
        self._prev_time = time.time()

    def stop(self) -> None:
        self.is_running = False
        if hasattr(self, 'processes'):
            [p.kill() for p in self.processes]

    def __del__(self) -> None:
        self.stop()

    def update_policy(self, policy: tuple[nn.Module] | str) -> None:
        assert not self.is_running
        self._policy = policy

    @property
    def total_steps(self) -> int:
        return self._total_steps.value

    @property
    def fps(self) -> float:
        prev_steps = self._prev_steps
        current_steps = self._prev_steps = self._total_steps.value
        prev_time = self._prev_time
        current_time = self._prev_time = time.time()
        fps = ((current_steps - prev_steps) / 
               (current_time - prev_time))
        return fps

    @property
    def average_return(self) -> float:
        avg_return = self._worker_return.value
        self._worker_return.value = 0.0
        return avg_return

    @staticmethod
    def _collect(replay_buffer: ReplayBuffer, policy: tuple[nn.Module] | str,
                 timesteps_per_sample: int, agent_update_freq: int, environment_args: Config,
                 agent_args: Config, step_counter: Synchronized, rb_lock: Lock) -> None:
        # Create data collector instance
        data_collector = DataCollector(replay_buffer, policy, timesteps_per_sample,
                                       agent_update_freq, environment_args, agent_args, rb_lock)
         # Setup infinite while-loop interupt handler
        global run_data_collection
        run_data_collection = True
        def signal_handler(signal, frame):
            global run_data_collection
            run_data_collection = False
        signal.signal(signal.SIGINT, signal_handler)
        # Loop forever...
        while run_data_collection:
            data_collector.step()
            step_counter.value += 1

    def _init_replay_buffer(self, replay_buffer: ReplayBuffer, 
                            policy: tuple[nn.Module] | str,
                            timesteps_per_sample: int,
                            agent_update_freq: int,
                            environment_args: Config,
                            agent_args: Config, rb_lock: Lock) -> None:
        dc = DataCollector(replay_buffer, policy, timesteps_per_sample,
                           agent_update_freq, environment_args, agent_args, rb_lock)
        self.action_repeat = dc.agent.action_repeat
        pbar = tqdm(total=timesteps_per_sample)
        while len(replay_buffer) == 0: 
            dc.step()
            self._total_steps.value += 1
        del dc

#####################################################################
#   Dataset
#####################################################################

class Dataset(IterableDataset):

    def __init__(self, batch_size: int,
                 replay_buffer: ReplayBuffer,
                 rb_lock: Lock) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.rb_lock = rb_lock

    def __iter__(self) -> Iterator:
        while True:
            with self.rb_lock:
                experience = self.replay_buffer.sample()
            for k,v in experience['observations'].items():
                if v.dtype == torch.uint8:
                    experience['observations'][k] = v.to(torch.float32)/255.
            yield experience

#####################################################################
#   Data-Module
#####################################################################

class DataModule(ptl.LightningDataModule):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size
        self.timesteps = cfg.timesteps

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # (1) Create Replay Buffer
            self.rb_lock = mp.Lock()
            self.replay_buffer = self.cfg.replay_buffer.create_instance(
                                    {'batch_size': self.batch_size,
                                     'collate_fn': self.collate_fn,
                                     'transform': self.stack_transform})
            self.replay_buffer._storage.scratch_dir = self.trainer.log_dir + '/replay-buffer'
            if self.cfg.has('replay_buffer_ckpt'):
                self.replay_buffer.loads(self.cfg.replay_buffer_ckpt)
            # (2) Create Data Collector
            self.data_collector = self.cfg.data_collector.create_instance({
                'replay_buffer': self.replay_buffer, 'policy': 'random',
                'timesteps_per_sample': self.timesteps, 'rb_lock': self.rb_lock})
            # (3) Prefill the replay buffer
            self.batch_agent_steps = self.batch_size * self.timesteps
            self.agent_action_repeat = self.data_collector.action_repeat 
            self.batch_env_steps = self.agent_action_repeat * self.batch_agent_steps
            if isinstance(self.data_collector, DataCollector):
                init_fill = max(self.batch_agent_steps, self.cfg.replay_buffer_start_capacity)
                init_fill = max(init_fill - len(self.replay_buffer), 0)
                for _ in tqdm(range(init_fill+1), desc='Prefilling Replay Buffer'):
                    self.data_collector.step()

    def teardown(self, stage: str) -> None:
        if (hasattr(self, 'data_collector') and
            hasattr(self.data_collector, 'stop')):
            self.data_collector.stop()

    @staticmethod
    def collate_fn(x: Any) -> Any:
        return x

    @staticmethod 
    def stack_transform(x: TensorDict|list[TensorDict]) -> TensorDict:
        if isinstance(x, list):
            return torch.stack(x)
        else:
            return x

    def train_dataloader(self) -> DataLoader:
        ds = Dataset(self.cfg.batch_size, self.replay_buffer, self.rb_lock)
        return DataLoader(ds, batch_size=None, collate_fn=self.collate_fn,
                          **self.cfg.dataloader_args.to_dict())
                        