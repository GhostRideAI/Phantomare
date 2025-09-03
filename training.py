# Imports
import math
import shutil
import copy
from enum import Enum
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from pytorch_lightning.callbacks import ProgressBar
import torch
from torch import Tensor
from tensordict import TensorDict
import torch.distributions as td
from torch.distributions import Distribution
from torchrl.modules import TruncatedNormal
import pytorch_lightning as ptl
from utils import Config, Utils 
from networks import Dreamer
from data import DataCollector, ConcurrentDataCollector

#from line_profiler import profile

#####################################################################
#   Lit-Module                                                 
#####################################################################

class LitModule(ptl.LightningModule):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg.to_dict())
        self.automatic_optimization = False
        self.advantage_ema = cfg.advantage_ema.create_instance()
        self.net = Dreamer(cfg.networks)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt_cfg = self.cfg.optimization
        wm_opt = opt_cfg.world_model.optimizer.create_instance(
            {'params': self.net.world_model.parameters()})
        actor_opt = opt_cfg.actor.optimizer.create_instance(
            {'params': self.net.actor.parameters()})
        critic_opt = opt_cfg.critic.optimizer.create_instance(
            {'params': self.net.critic.parameters()})
        return wm_opt, actor_opt, critic_opt
    
    def configure_callbacks(self) -> ptl.Callback:
       return [self.CustomProgressBar(),
               self.LoggingCallback(**self.cfg.logging_callback_args.to_dict())]

    def on_fit_start(self) -> None:
        # Create a copy of the config file
        if hasattr(self.cfg, 'file') and hasattr(
            self, 'logger') and hasattr(self.logger, 'log_dir'):
            shutil.copy(self.cfg.file, self.logger.log_dir)
    
    def on_train_start(self) -> None:
        # Calcuate train ratio stuff
        batch_agent_steps = self.trainer.datamodule.batch_agent_steps
        replay_ratio = self.cfg.replay_ratio
        self.agent_steps_per_grad_step = batch_agent_steps // replay_ratio
        self._next_grad_step = self.agent_steps_per_grad_step

    #@profile
    def on_train_batch_start(self, *args, **kwargs) -> None:
        # Update Target Critic
        update_freq = self.cfg.optimization.target_critic.update_freq
        update_tau = self.cfg.optimization.target_critic.update_tau
        if (self.global_step % update_freq) == 0:
            self.net.update_target_critic(update_tau)
        # Update Data Collection Agent
        update_freq = self.trainer.datamodule.data_collector.agent_update_freq
        if (self.global_step % update_freq) == 0:
            self.trainer.datamodule.data_collector.update_policy(
                (copy.deepcopy(self.net.world_model).to('cpu', non_blocking=True),
                 copy.deepcopy(self.net.actor).to('cpu', non_blocking=True)))
        # Step Agent in Real Environment
        if isinstance(self.trainer.datamodule.data_collector, DataCollector):
            for _ in range(self.agent_steps_per_grad_step):
                self.trainer.datamodule.data_collector.step()
        elif isinstance(self.trainer.datamodule.data_collector, ConcurrentDataCollector) and (
                        not self.trainer.datamodule.data_collector.decoupled):
            agent_steps = self.trainer.datamodule.data_collector.total_steps
            while agent_steps < self._next_grad_step:
                agent_steps = self.trainer.datamodule.data_collector.total_steps
            self._next_grad_step += self.agent_steps_per_grad_step
        # If time, Perform Evaluation of Agent
        if isinstance(self.trainer.datamodule.data_collector, DataCollector):
            if (self.global_step % self.cfg.eval_every_n_steps) == 0:
                    self.trainer.datamodule.data_collector.evaluate_agent(self.cfg.eval_n_episodes)

    #@profile
    def training_step(self, batch: TensorDict, batch_idx: int) -> dict[str, Tensor]:
        # (0) Prepare for Step
        observations, actions, continues = (
            batch['observations'], batch['actions'], batch['continues'])
        world_model_opt, actor_opt, critic_opt = self.optimizers()
        # HACK: https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        world_model_opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
        world_model_opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        actor_opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
        actor_opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        # (1) Update World-Model
        world_model_output = self.net.world_model(observations, actions, continues)
        world_model_losses = self.world_model_loss_func(world_model_output, batch)
        world_model_opt.zero_grad(); self.manual_backward(world_model_losses['dynamics/loss'])
        if self.cfg.optimization.world_model.has('gradient_clip'):
            self.clip_gradients(world_model_opt, **self.cfg.optimization.world_model.gradient_clip.to_dict())
        world_model_opt.step()
        # (2) Dream a Scenario
        posteriors, recurrents = [t.detach().flatten(0,1) for t in world_model_output[-2:]]
        dream = self.net.world_model.dream(posteriors, recurrents, self.net.actor,
                                           self.cfg.optimization.behavior.dream_horizon)
        # (3) Update Actor & Critic From Dream
        states, _, continues, action_distribution, reward_distribution, _ = dream
        rewards = reward_distribution.mode
        values = self.net.critic(states).mode
        # (3.1) Calculate Lambda Returns
        return_lambda = self.cfg.optimization.behavior.return_lambda
        discount_gamma = self.cfg.optimization.behavior.discount_gamma
        lambda_returns = self._compute_lambda_return(
            rewards, values, continues, return_lambda, discount_gamma)
        discounts = torch.nan_to_num(torch.cumprod(
            continues.detach() * discount_gamma, dim=1) / discount_gamma)
        values = values[:,:-1]; states = states[:,:-1].detach(); discounts = discounts[:,:-1]
        # (3.2) Update Actor
        actor_loss, advantage = self.actor_loss_func(
            action_distribution, values, lambda_returns, discounts)
        actor_opt.zero_grad(); self.manual_backward(actor_loss)
        if self.cfg.optimization.actor.has('gradient_clip'):
            self.clip_gradients(actor_opt, **self.cfg.optimization.actor.gradient_clip.to_dict())
        actor_opt.step()
        # (3.3) Update Critic
        value_distribution = self.net.critic(states)
        target_values = self.net.target_critic(states).mode
        critic_loss = self.critic_loss_func(
            value_distribution, target_values.detach(),
            lambda_returns.detach(), discounts.detach())
        critic_opt.zero_grad(); self.manual_backward(critic_loss)
        if self.cfg.optimization.critic.has('gradient_clip'):
            self.clip_gradients(critic_opt, **self.cfg.optimization.critic.gradient_clip.to_dict())
        critic_opt.step()
        # (4) Return Metrics for Logging
        # HACK: for torch.compile
        if 'world_model_losses' in locals():
            results = self._create_log_dict(locals())
        else:
            results = {}
        return results

    def world_model_loss_func(self, wm_outputs: tuple[Tensor], batch: TensorDict) -> Tensor:
        # Unpack
        loss_cfg = self.cfg.optimization.dynamics
        observations, rewards, continues = (
            batch['observations'], batch['rewards'], batch['continues'])
        (prior_latent_dist, posterior_latent_dist,
         est_obs, est_reward_dist, est_continue_dist, _, _) = wm_outputs
        # Prediction Loss
        reconstruction_loss = 0.0
        for name, dist in est_obs.items():
            dist = dist.data
            reconstruction_loss += -dist.log_prob(observations[name])
        reward_loss = -est_reward_dist.log_prob(rewards)
        continue_loss = -est_continue_dist.log_prob(continues.float())
        prediction_loss = reconstruction_loss + reward_loss + continue_loss
        # Dynamics Loss
        dynamics_loss = td.kl_divergence(Utils.one_hot_cat_dist(
            posterior_latent_dist.base_dist.logits.detach()),
            prior_latent_dist).clamp(1.0)
        # Representation Loss
        representation_loss = td.kl_divergence(
            posterior_latent_dist, Utils.one_hot_cat_dist(
            prior_latent_dist.base_dist.logits.detach())
            ).clamp(1.0)
        # Combined Loss
        loss = (loss_cfg.prediction_loss_weight * prediction_loss +
                loss_cfg.dynamics_loss_weight * dynamics_loss +
                loss_cfg.representation_loss_weight * representation_loss).mean()
        # Return all losses
        loss_dict = {
            'dynamics/loss': loss,
            'dynamics/reconstruction_loss': reconstruction_loss.mean(),
            'dynamics/reward_loss': reward_loss.mean(),
            'dynamics/continue_loss': continue_loss.mean(),
            'dynamics/prediction_loss': prediction_loss.mean(),
            'dynamics/dynamics_loss': dynamics_loss.mean(),
            'dynamics/representation_loss': representation_loss.mean()}
        return loss_dict

    def actor_loss_func(self, actions_dist: TruncatedNormal, values: Tensor,
                        lambda_returns: Tensor, discounts: Tensor) -> Tensor:
        baseline = values
        offset, invscale = self.advantage_ema(lambda_returns)
        lambda_returns = (lambda_returns - offset) / invscale
        baseline = (baseline - offset) / invscale
        advantage = lambda_returns - baseline
        entropy = actions_dist.entropy().unsqueeze(-1)[:,:-1]
        entropy_weight = self.cfg.optimization.behavior.action_entropy_weight
        actor_loss = -torch.mean(discounts * (advantage + entropy_weight * entropy))
        return actor_loss, advantage.mean()

    def critic_loss_func(self, value_dist: Distribution,
                          target_values: Tensor, lambda_returns: Tensor,
                          discounts: Tensor) -> Tensor: 
        critic_loss = (-value_dist.log_prob(lambda_returns) -
                       value_dist.log_prob(target_values))
        critic_loss = torch.mean(critic_loss * discounts.squeeze(-1))
        return critic_loss

    def _compute_lambda_return(self, rewards: Tensor,
                               values: Tensor, continues: Tensor,
                               lmbda: float, gamma: float) -> Tensor:
        rewards = rewards[:,:-1]
        continues = gamma * continues[:,:-1]
        vals = [values[:,-1]]
        interm = rewards + continues * values[:,1:] * (1 - lmbda)
        for t in reversed(range(interm.size(1))):
            vals += [ interm[:,t] + continues[:,t] * lmbda * vals[-1] ]
        returns = torch.cat(list(reversed(vals))[:-1], 1)
        return returns.unsqueeze(-1)
    
    def _create_log_dict(self, locs: dict) -> dict:
        results = {cat: {} for cat in ('metrics', 'hists', 'imgs', 'vids')}
        results['metrics'].update({k:v.cpu() for k,v in locs['world_model_losses'].items()})
        results['metrics']['behavior/critic_loss'] = locs['critic_loss'].cpu()
        results['metrics']['behavior/actor_loss'] = locs['actor_loss'].cpu()
        # Handle debugging logs
        if self.global_step % self.trainer.log_every_n_steps == 0:
            # (0) Batch from real-env.
            batch = locs['batch']
            for name, obs in batch['observations'].items():
                if name == 'image':
                    results['vids']['environment/'+f'{name}_observations'] = obs.detach().cpu()
                else:
                    results['hists']['environment/'+f'{name}_observations'] = obs.detach().cpu()
            for i in range(batch['actions'].size(-1)):
                results['hists'][f'environment/action_{i}'] = batch['actions'][:,:,i].cpu()
            results['hists']['environment/rewards'] = batch['rewards'].cpu()
            results['hists']['environment/continues'] = batch['continues'].cpu()
            # (1) World-Model
            (prior_dist, posterior_dist, est_obs,
             est_reward_dist, est_continue_dist, _, _) = locs['world_model_output']
            results['hists']['dynamics/prior'] = prior_dist.base_dist.probs.cpu()
            results['metrics']['dynamics/prior_entropy'] = prior_dist.base_dist.entropy().mean().cpu()
            results['hists']['dynamics/posterior'] = posterior_dist.base_dist.probs.cpu()
            results['metrics']['dynamics/posterior_entropy'] = posterior_dist.base_dist.entropy().mean().cpu()
            if 'image' in est_obs.keys():
                results['vids']['dynamics/estimated_image_observations'] = est_obs['image'].mode.detach().cpu()
            results['hists']['dynamics/estimated_rewards'] = est_reward_dist.mode.cpu()
            results['hists']['dynamics/estimated_continues'] = est_continue_dist.mode.cpu()
            # (2) Dream
            states, actions, conts, _, reward_dist, _ = locs['dream']
            if hasattr(self.net.world_model, 'image_decoder'):
                self.net.world_model.image_decoder.eval()
                with torch.no_grad():
                    steps = states.size(1)
                    results['vids']['dream/image_observations'] = self.net.world_model.image_decoder(
                        states[None,0].flatten(0,1)).mode.unflatten(0, (1,steps)).detach().cpu()
                del states; self.net.world_model.image_decoder.train()
            for i in range(actions.size(-1)):
                results['hists'][f'dream/action_{i}'] = actions[:,:,i].cpu()
            results['hists']['dream/rewards'] = reward_dist.mode.cpu()
            results['metrics']['dream/return'] = reward_dist.mode.sum(1).mean().cpu()
            results['metrics']['dream/action_entropy'] = (
                locs['action_distribution'].entropy().mean().cpu())
            results['hists']['dream/continues'] = conts.cpu()
            # (3) Actor-Critic
            results['hists']['behavior/lambda_returns'] = locs['lambda_returns'].cpu()
            results['hists']['behavior/critic_value_estimate'] = locs['values'].cpu()
            results['metrics']['behavior/advantage'] = locs['advantage'].cpu()
        return results

    class CustomProgressBar(ProgressBar):
        """
        Custom PTL progress bar which reports important info.
        This gets added to the PTL Trainer's callbacks in the
        configure_callbacks method in the LitModule.
        """
        def __init__(self) -> None:
            super().__init__()
            self.enable = True

        def disable(self) -> None:
            self.enable = False

        def on_train_batch_end(self, trainer: ptl.Trainer, pl_module: ptl.LightningModule, *args, **kwargs):
            super().on_train_batch_end(trainer, pl_module, *args, **kwargs)
            if 'environment/return' in trainer.callback_metrics:
                env_return = trainer.callback_metrics['environment/return'].item()
            else:
                env_return = -math.inf
            print(f"👻 > Gradient-Step: {pl_module.global_step} || "
                  f"Environment-Step: {trainer.datamodule.data_collector.total_steps} || "
                  f"Environment-Return: {env_return:.3f}", end='\r')
    
    class LoggingCallback(ptl.Callback):
        
        class LogLevel(Enum):
            OFF = 0
            NORMAL = 1
            DEBUG = 2
        
        def __init__(self, log_level: int=0) -> None:
            super().__init__()
            self.log_level = self.LogLevel(log_level)
            self.current_step_logs = {}

        def on_train_batch_end(self, trainer: ptl.Trainer,
                               pl_module: ptl.LightningModule,
                               outputs: dict[str, dict[str, Tensor]],
                               batch: TensorDict, batch_idx: int) -> None:
            if pl_module.logger is None: return
            if self.log_level.value > 0:
                for name, metric in outputs['metrics'].items():
                    pl_module.log(name, metric)
                for name, vid in outputs['vids'].items():
                    vid = Utils.minmax_scale(vid.float()[0].unsqueeze(0)).cpu().numpy()
                    if vid.shape[-3] == 1: vid = vid.repeat(3,2)
                    pl_module.logger.experiment.add_video(
                        name, vid, global_step=trainer.global_step, fps=5)
                for name, tensor in outputs['hists'].items():
                    tensor = tensor[0]
                    try:
                        pass #NOTE: sloooow!
                        #pl_module.logger.experiment.add_histogram(
                        #    name, tensor, pl_module.global_step)
                    except: pass
            if (self.log_level == self.LogLevel.DEBUG and
                (batch_idx % trainer.log_every_n_steps) == 0):
                #self._log_network_metrics(pl_module)
                rb_len = len(trainer.datamodule.replay_buffer)
                pl_module.log('environment/replay_buffer_length', rb_len)
                avg_return = trainer.datamodule.data_collector.average_return
                pl_module.log('environment/return', avg_return, prog_bar=True)
                pl_module.log('hp_metric', avg_return)
                agent_steps = trainer.datamodule.data_collector.total_steps
                action_repeat = trainer.datamodule.data_collector.action_repeat
                fps = trainer.datamodule.data_collector.fps
                pl_module.log('environment/agent_steps', agent_steps)
                pl_module.log('environment/environment_steps', agent_steps * action_repeat)
                pl_module.log('environment/FPS', fps * action_repeat)

        def _log_network_metrics(self, pl_module: ptl.LightningModule) -> None:
            grad_avgs = {}
            for n, p in pl_module.named_parameters():
                if n.split('.')[1] == 'world_model': n = 'dynamics/' + n
                else: n = 'behavior/' + n
                if p.requires_grad:
                    try:
                        pl_module.logger.experiment.add_histogram(
                            n, p, pl_module.global_step)
                    except: pass
                    if p.grad is not None:
                        try:
                            pl_module.logger.experiment.add_histogram(
                                n+'.grad', p.grad, pl_module.global_step)
                            n = n.split('.')[:2]
                            n = '.'.join(n)
                            if n in grad_avgs:
                                grad_avgs[n] = (grad_avgs[n] + p.grad.mean()) / 2
                            else:
                                grad_avgs[n] = p.grad.mean()
                        except: pass
            for n, grad_avg in grad_avgs.items():
                pl_module.log(n + ' gradient_average', grad_avg)
