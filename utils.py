# Imports
import warnings
import copy
import importlib
import json
import multiprocessing as mp
from types import SimpleNamespace
from typing import Any
from pathlib import Path
import yaml
import torch
from torch import Tensor, nn
import torch.distributions as td
import torchvision as tv
from tensordict import TensorDict
from torchrl.data import Storage
import numpy as np
from numpy.typing import NDArray
import imageio

#####################################################################
#   Misc. Utils
#####################################################################
 
class Utils:
    ten2img = tv.transforms.ToPILImage()
    img2ten = tv.transforms.PILToTensor()
    totensor = tv.transforms.ToTensor()

    def minmax_scale(t: Tensor) -> Tensor:
        return (t - t.min())/(t.max() - t.min())

    def symlog(x: Tensor) -> Tensor:
        return torch.sign(x) * torch.log(1 + torch.abs(x))

    def symexp(x: Tensor) -> Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
    
    def one_hot_cat_dist(l: Tensor) -> td.Distribution:
        return td.Independent(td.OneHotCategoricalStraightThrough(logits=l), 1)

    def uniform_mix(logits: Tensor, unimix: float = 0.01) -> Tensor:
        if unimix > 0.0:
            probs = logits.softmax(dim=-1)
            uniform = 1 / probs.size(-1)
            probs = (1 - unimix) * probs + unimix * uniform
            logits = td.utils.probs_to_logits(probs)
        return logits

    def create_mp4(images: list[NDArray], filepath: Path, fps: int) -> None:
        with imageio.get_writer(filepath, fps=fps) as writer:
            for img in images: writer.append_data(img)

    def create_gif(images: list[NDArray], filepath: Path, fps: float) -> None:
        with imageio.get_writer(filepath, mode='I', fps=fps, loop=0) as writer:
            for img in images: writer.append_data(img)

    class Symlog(nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, x: Tensor) -> Tensor:
            return Utils.symlog(x)

    class UniformMix(nn.Module):
        def __init__(self, unimix: float = 0.01) -> None:
            super().__init__()
            self.unimix = unimix

        def forward(self, logits: Tensor) -> Tensor:
            return Utils.uniform_mix(logits, self.unimix)

#####################################################################
#   Configuration
#####################################################################

class Config(SimpleNamespace):

    debug: bool = False
    file: Path = None

    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
          return Config(**entry)
        return entry

    @staticmethod
    def rev_map_entry(entry):
        if isinstance(entry, Config):
          return entry.to_dict_recursive()
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, Config(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

    def to_dict(self) -> dict:
        # Convert the Config to a dictionary
        return vars(self)
    
    def to_dict_recursive(self) -> dict:
        # Convert all Configs contained in this Config to dictionaries
        self = self.to_dict()
        for key, val in self.items():
            if isinstance(val, Config):
                self[key] = val.to_dict_recursive()
            elif isinstance(val, list):
                self[key] = list(map(Config.rev_map_entry, val))
        return self

    def create_instance(self, additional_kwargs: dict=None) -> object:
        # Convert to dictionary
        cfg = self.to_dict()
        # Load the module class
        Class = self._get_module_class(next(iter(cfg)))
        # Get argments to provide to class
        kwargs = copy.deepcopy(next(iter(cfg.values())).to_dict())
        # Check arguments for classes/instances that need to be loaded
        for k,v in kwargs.items():
            if isinstance(v, str) and '.' in v and '/' not in v:
                kwargs[k] = self._get_module_class(v)
            elif isinstance(v, Config) and '.' in next(iter(v.to_dict().keys())):
                kwargs[k] = v.create_instance()
            elif isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], str) and '.' in v[i] and '/' not in i:
                        v[i] = self._get_module_class(v[i])
                    elif isinstance(v[i], Config) and '.' in next(iter(v[i].to_dict().keys())):
                        v[i] = v[i].create_instance()
        # If additional/default arguments were provided from source code...
        if additional_kwargs:
            # Scan the additional arguments
            for k,v in additional_kwargs.items():
                # If they are already defined by user, warn
                if k in kwargs:
                    warnings.warn(f'During creation of {Class} instance, '
                                  f'default argument of {k} is being overriden by user. '
                                  f'Ensure this is what you intended!')
                # Else update arguments dictionary with additional argument
                else:
                    kwargs[k] = v
        return Class(**kwargs)
    
    def load_all_instances(self) -> None:
        for k,v in self.to_dict().items():
            if isinstance(v, str) and '.' in v and '/' not in v:
                self.to_dict()[k] = self._get_module_class(v)
            elif isinstance(v, Config) and '.' in next(iter(v.to_dict().keys())):
                self.to_dict()[k] = v.create_instance()
            elif isinstance(v, list):
                for i in range(len(v)):
                    if isinstance(v[i], str) and '.' in v[i] and '/' not in i:
                        v[i] = self._get_module_class(v[i])
                    elif isinstance(v[i], Config) and '.' in next(iter(v[i].to_dict().keys())):
                        v[i] = v[i].create_instance()

    def _get_module_class(self, spec: str) -> Any:
        module_and_class = spec.split('.')
        clss = module_and_class[-1]
        mod = '.'.join(module_and_class[:-1])
        for _ in range(2): #WORKAROUND FOR NeMO EMA...(ugh)
            try: mod = importlib.import_module(mod); break
            except: pass
        Class = getattr(mod, clss)
        return Class

    def has(self, attr: str) -> bool:
        if (hasattr(self, attr) and
            getattr(self, attr) is not None):
            return True
        else:
            return False
 
    @staticmethod
    def from_yaml(config_file: Path|str):
        if isinstance(config_file, str):
            config_file = Path(config_file)
        with open(config_file) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = Config(**config_dict, file=config_file)
        return cfg
    
    @staticmethod
    def from_yaml_string(yaml_str: str):
        cfg = Config(**yaml.safe_load(yaml_str))
        return cfg

#####################################################################
#   EMA
#####################################################################

class EMA(nn.Module):

    def __init__(self, decay: float=0.99,
                 maximum: float=1.0,
                 percentile_low: float=0.05,
                 percentile_high=0.95) -> None:
        super().__init__()
        self.decay = decay
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.register_buffer('max', torch.as_tensor(maximum))
        self.register_buffer('low', torch.zeros((), dtype=torch.float32))
        self.register_buffer('high', torch.zeros((), dtype=torch.float32))
    
    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor]:
        low = torch.quantile(x, self.percentile_low)
        high = torch.quantile(x, self.percentile_high)
        self.low = self.decay * self.low + (1 - self.decay) * low
        self.high = self.decay * self.high + (1 - self.decay) * high
        invscale = torch.max(1/self.max, self.high - self.low)
        return self.low, invscale

#####################################################################
#   Compressed Replay Buffer Storage
#####################################################################

class CompressedStorage(Storage):

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self.scratch_dir = '/tmp/replay-buffer'
        self._n_items = mp.Value('I', 0)

    @property
    def scratch_dir(self) -> Path:
        return self._scratch_dir
    
    @scratch_dir.setter
    def scratch_dir(self, value: str | Path) -> None:
        self._scratch_dir = Path(value) / 'storage'

    def set(self, indices: int|list[int], data: TensorDict|list[TensorDict]) -> None:
        if isinstance(indices, int): indices = [indices]; data = [data]
        for idx in range(len(data)):
            self._single_set(indices[idx], data[idx])

    def _single_set(self, index: int, data: TensorDict) -> None:
        data = {k: v.cpu().numpy() for k,v in data.flatten_keys().items()}
        self.scratch_dir.mkdir(parents=True, exist_ok=True)
        filename = self.scratch_dir / (str(index).zfill(len(str(self.max_size))))
        np.savez_compressed(filename, **data)
        self._n_items.value = min(self._n_items.value + 1, self.max_size)
        if self._n_items.value < self.max_size: self.dumps(self.scratch_dir)
    
    def get(self, indices: int|list[int]) -> TensorDict|list[TensorDict]:
        if isinstance(indices, int): indices = [indices]
        data = [self._single_get(i.item()) for i in indices]
        data = torch.stack(data)
        return data

    def _single_get(self, index: Tensor) -> TensorDict:
        filename = self.scratch_dir / (str(index).zfill(
            len(str(self.max_size))) + '.npz')
        data = np.load(filename).items()
        data = {k: torch.from_numpy(v) for k,v in data}
        batch_size = next(iter(data.values())).size(0)
        data = TensorDict(data, [batch_size]).unflatten_keys()
        return data
    
    def dumps(self, path: Path|str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "storage_metadata.json", "w") as file:
            json.dump({'len': len(self)}, file)
    
    def loads(self, path: Path|str) -> None:
        self.scratch_dir = path
        with open(path / "storage_metadata.json", "r") as file:
            metadata = json.load(file)
        self._n_items.value = metadata['len']
    
    def __len__(self) -> int:
        return self._n_items.value
