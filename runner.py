# Imports
import os
import copy
import argparse
import random
import numpy as np
import torch
import pytorch_lightning as ptl
from pytorch_lightning.loggers import CSVLogger
from ray.train.lightning import prepare_trainer
from ray.train.torch import TorchTrainer
from ray import tune
import torch_pruning as tp
from data import DataModule
from training import LitModule
from utils import Config

#####################################################################
#   Runners
#####################################################################

class Runner:
    """
    Handles setting up all various sub-modules for training.
    Args:
        global_cfg: A config with all sub-modules configs
            (runner, lit_module, data_module)
        _using_ray: Internal argument. Set to True when using Ray to
            prepare the PTL trainer.
        _override_exp_dir: Internal argument. Config containing save_dir,
            name, & version to override logger settings.
    """
    def __init__(self, global_cfg: Config,
                 _using_ray: bool=False,
                 _override_exp_dir: Config=None) -> None:
        self._using_ray = _using_ray
        self.global_cfg = global_cfg
        self.cfg = global_cfg.runner
        self.lm_cfg = global_cfg.lit_module
        self.dm_cfg = global_cfg.data_module
        self.cfg.file = self.global_cfg.file
        self.lm_cfg.file = self.global_cfg.file
        self.dm_cfg.file = self.global_cfg.file
        self._override_exp_dir = _override_exp_dir
        self._setup()

    def execute(self, routine: str='train') -> None:
        if routine == 'train':
            if not self.global_cfg.debug and self.lm_cfg.compile:
                self._lit_module.net = torch.compile(self._lit_module.net)
                self._lit_module.advantage_ema = torch.compile(self._lit_module.advantage_ema)
            self._ptl_trainer.fit(self._lit_module, self._data_module,
                                  ckpt_path=(
                                      self.cfg.resume_training_ckpt if
                                      self.cfg.has('resume_training_ckpt') else None))
        else:
            raise NotImplementedError(routine)

    def _setup(self) -> None:
        self._setup_data_module()
        self._setup_lit_module()
        self._setup_ptl_trainer()

    def _setup_lit_module(self, save_hparams: bool = True) -> None:
        if self.cfg.has('starting_params_ckpt'):
            self._lit_module = LitModule.load_from_checkpoint(
                self.cfg.starting_params_ckpt, cfg=self.lm_cfg, strict=False)
        else:
            self._lit_module = LitModule(self.lm_cfg)
        cfg = copy.deepcopy(self.global_cfg).to_dict_recursive()
        if save_hparams:
            self._lit_module.save_hyperparameters(cfg)

    def _setup_data_module(self) -> None:
        self._data_module = DataModule(self.dm_cfg)

    def _setup_ptl_trainer(self) -> None:
        self._prep_loggers()
        self.cfg.ptl_trainer_args.load_all_instances()
        if self._using_ray:
            self._ptl_trainer = ptl.Trainer(
                    num_sanity_val_steps=0,
                    enable_progress_bar=False,
                    enable_checkpointing=False,
                    **self.cfg.ptl_trainer_args.to_dict())
            self._ptl_trainer = prepare_trainer(self._ptl_trainer)
        else:
            self._ptl_trainer = ptl.Trainer(
                    num_sanity_val_steps=0,
                    **self.cfg.ptl_trainer_args.to_dict())
    
    def _prep_loggers(self) -> None:
        if self.cfg.ptl_trainer_args.has('logger'):
            if isinstance(self.cfg.ptl_trainer_args.logger, list):
                for i in range(len(self.cfg.ptl_trainer_args.logger)):
                    if self._override_exp_dir:
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].save_dir = self._override_exp_dir.save_dir
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].name = self._override_exp_dir.name
                        list(self.cfg.ptl_trainer_args.logger[
                            i].to_dict().values())[0].version = self._override_exp_dir.version
                    self.cfg.ptl_trainer_args.logger[i] = self.cfg.ptl_trainer_args.logger[i].create_instance()
                l = self.cfg.ptl_trainer_args.logger[0]
                self.cfg.ptl_trainer_args.logger += [CSVLogger(l.save_dir, l.name, l.version)]
            else:
                if self._override_exp_dir:
                    list(self.cfg.ptl_trainer_args.logger.save_dir.to_dict().values())[0] = self._override_exp_dir.save_dir
                    list(self.cfg.ptl_trainer_args.logger.name)[0] = self._override_exp_dir.name
                    list(self.cfg.ptl_trainer_args.logger.version)[0] = self._override_exp_dir.version
                self.cfg.ptl_trainer_args.logger = self.cfg.ptl_trainer_args.logger.create_instance()
                l = self.cfg.ptl_trainer_args.logger
                self.cfg.ptl_trainer_args.logger = [l, CSVLogger(l.save_dir, l.name, l.version)]
            self._log_dir = Config(save_dir=l.save_dir, name=l.name, version=(
                f'version_{l.version}' if isinstance(l.version, int) else l.version))

class RayRunner:
    """
    A wrapper around Runner to leverage the Ray API (e.g. Tune, TorchTrainer).
    Args:
        global_cfg: A config with all sub-modules configs
            (runner, lit_module, data_module)
    """
    def __init__(self, global_cfg: Config) -> None:
        self._tuning = False
        if global_cfg.runner.has('ray_tuner_args'):
            self._tuning = True
        self.cfg = copy.deepcopy(global_cfg.runner)
        self.global_cfg = global_cfg
        self._setup()

    def execute(self, routine: str) -> None:
        if routine != 'train':
            raise NotImplementedError(routine)
        if self._tuning:
            self._tuner.fit()
        else:
            self._ray_trainer.fit()

    def _setup(self) -> None:
        os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'
        self._prepare_global_cfg()
        self._setup_ray_trainer()
        if self._tuning:
            self._setup_tuner()

    def _setup_tuner(self) -> None:
        self.cfg.ray_tuner_args.load_all_instances()
        self._tuner = tune.Tuner(
            self._ray_trainer, param_space={
                "train_loop_config": self.global_cfg.to_dict_recursive()},
            **self.cfg.ray_tuner_args.to_dict())

    def _setup_ray_trainer(self) -> None:
        self.cfg.ray_trainer_args.load_all_instances()
        self._ray_trainer = TorchTrainer(
            self._train_fn,
            train_loop_config=self.global_cfg,
            **self.cfg.ray_trainer_args.to_dict())

    def _prepare_global_cfg(self) -> None:
        def search(cfg):
            if isinstance(cfg, Config):
                cfg = cfg.to_dict()
            for k,v in cfg.items():
                if isinstance(v, Config):
                    search(v)
                elif isinstance(v, list):
                    for i in range(len(v)):
                        if isinstance(v[i], Config):
                            search(v[i])
                        elif isinstance(v[i], str) and v[i].startswith('tune.'):
                                assert self._tuning, (
                                    'tuning requires ray_tuner_args to be defined!')
                                v[i] = eval(v[i])
                elif isinstance(v, str) and v.startswith('tune.'):
                    assert self._tuning, (
                        'tuning requires ray_tuner_args to be defined!')
                    cfg[k] = eval(v)
        search(self.global_cfg)

    def _train_fn(self, cfg: Config|dict) -> None:
        if isinstance(cfg, dict):
            cfg = Config(**cfg)
        runner = Runner(cfg, _using_ray=True)
        runner.execute()

class PruningRunner(Runner):
    """
    An extension of the Runner class to support network pruning via TorchPruning.
    (https://github.com/VainF/Torch-Pruning)
    """
    def __init__(self, global_cfg: Config, _using_ray: bool = False) -> None:
        self._og_cfg = copy.deepcopy(global_cfg)
        super().__init__(global_cfg, _using_ray)

    def _setup(self) -> None:
        super()._setup_data_module()
        super()._setup_lit_module(save_hparams=False)
        super()._setup_ptl_trainer()
        self._setup_pruner()

    def _setup_pruner(self) -> None:
        ignored_layers = self._lit_module.net.backbone.out_layers
        self._example_input = {'x': self._lit_module.example_input_array['video'].flatten(0,1)}
        additional_args = {'model': self._lit_module.net.backbone,
                           'example_inputs': self._example_input,
                           'ignored_layers': ignored_layers}
        self._pruner = self.cfg.pruner.create_instance(additional_args)
 
    def execute(self, routine: str = 'train') -> None:
        _, start_nparams = tp.utils.count_ops_and_params(
            self._lit_module.net.backbone,
            self._example_input)
        for i in range(self._pruner.iterative_steps):
            _, nparams = tp.utils.count_ops_and_params(
                self._lit_module.net.backbone,
                self._example_input)
            print(f' {(1 - (nparams/start_nparams)):.2%} Pruned '.center(70, '*'))
            super().execute(routine)
            if i < (self._pruner.iterative_steps - 1):
                # Perform pruning
                self._pruner.step()
                # Calculate new number of parameters
                _, nparams = tp.utils.count_ops_and_params(
                    self._lit_module.net.backbone,
                    self._example_input)
                # Calculate pruned amount
                pruned_amt = f'{(1 - (nparams/start_nparams)):.2%}-pruned'
                # Override experiment directory for new pruned version
                self._override_exp_dir = self._log_dir
                self._override_exp_dir.version = (
                    f'{self._override_exp_dir.version.split("/")[0]}/{pruned_amt}')
                # Load a new copy of the original config
                global_cfg = copy.deepcopy(self._og_cfg)
                self.global_cfg = global_cfg
                self.cfg = global_cfg.runner
                self.lm_cfg = global_cfg.lit_module
                self.dm_cfg = global_cfg.data_module
                # Re-setup data module & PTL trainer
                super()._setup_data_module()
                super()._setup_ptl_trainer()

def main() -> None:
    # CLI
    parser = argparse.ArgumentParser(description='Run training, etc.')
    parser.add_argument('--config', type=str, required=True, help='Path to .yaml configuration file.')
    parser.add_argument('--routine', type=str, required=True, help='What routine to run. (Options: train)')
    parser.add_argument('--debug', action='store_true', help='Flag to run in debug mode.')
    args = parser.parse_args()

    setattr(Config, 'debug', args.debug)
    cfg = Config.from_yaml(args.config)
    seed = cfg.runner.rng_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if (cfg.runner.has('type') and cfg.runner.type == 'ray'):
        assert cfg.runner.has('ray_trainer_args'), (
            'ray_trainer_args must be defined in config '
            'to user runner type == \'ray\'')
        runner = RayRunner(cfg)
    elif (cfg.runner.has('type') and cfg.runner.type == 'prune'):
        assert cfg.runner.has('pruner'), (
            'pruner must be defined in config '
            'to user runner type == \'prune\'')
        runner = PruningRunner(cfg)
    else:
        runner = Runner(cfg)
    runner.execute(args.routine)


if __name__ == '__main__':
    main()

