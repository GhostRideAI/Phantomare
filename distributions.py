# Imports
from typing import Callable
import torch
from torch import Tensor, nn
from torch.nn import functional as F
import torch.distributions as td
from torch.distributions import Distribution
from torchrl.modules import TruncatedNormal
from utils import Utils

#####################################################################
#   Distribution Modules
#####################################################################

class TruncNormalDistribution(nn.Module):

    def __init__(self, min_scale: float = 0.1,
                 max_scale: float = 1.0,
                 min_value: float = -1.0,
                 max_value: float = 1.0) -> None:
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, logits: Tensor = None, loc: Tensor = None,
                scale: Tensor = None) -> TruncatedNormal:
        if logits is not None:
            loc, scale = logits.chunk(2, -1)
            loc = loc.tanh()
            scale = ((self.max_scale - self.min_scale) *
                     torch.sigmoid(scale) + self.min_scale)
        dist = TruncatedNormal(loc.contiguous(), scale.contiguous(),
                               low=self.min_value, high=self.max_value)
        return dist


class TwoHotDistribution(nn.Module):

    def __init__(self, dims: int = 1,
                 low: int = -20, high: int = 20,
                 transfwd: Callable = Utils.symlog,
                 transbwd: Callable = Utils.symexp) -> None:
        super().__init__()
        self.dims = dims
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd

    def forward(self, logits: Tensor) -> Distribution:
        return self.Distribution(
            logits, self.dims, self.low,
            self.high, self.transfwd, self.transbwd)
    
    class Distribution:

        def __init__(self, logits: Tensor, dims: int = 1,
                     low: int = -20, high: int = 20,
                     transfwd: Callable[[Tensor], Tensor] = Utils.symlog,
                     transbwd: Callable[[Tensor], Tensor] = Utils.symexp) -> None:
            self.logits = logits
            self.probs = F.softmax(logits, dim=-1)
            self.dims = tuple([-x for x in range(1, dims + 1)])
            self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)
            self.low = low
            self.high = high
            self.transfwd = transfwd
            self.transbwd = transbwd
            self._batch_shape = logits.shape[: len(logits.shape) - dims]
            self._event_shape = logits.shape[len(logits.shape) - dims : -1] + (1,)

        @property
        def mean(self) -> Tensor:
            return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

        @property
        def mode(self) -> Tensor:
            return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))

        def log_prob(self, x: Tensor) -> Tensor:
            x = self.transfwd(x)
            below = (self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1
            above = len(self.bins) - (self.bins > x).type(torch.int32).sum(dim=-1, keepdim=True)
            below = torch.clip(below, 0, len(self.bins) - 1)
            above = torch.clip(above, 0, len(self.bins) - 1)
            equal = below == above
            dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
            dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
            total = dist_to_below + dist_to_above
            weight_below = dist_to_above / total
            weight_above = dist_to_below / total
            target = (
                F.one_hot(below, len(self.bins)) * weight_below[..., None]
                + F.one_hot(above, len(self.bins)) * weight_above[..., None]
            ).squeeze(-2)
            log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
            return (target * log_pred).sum(dim=self.dims)


class SymlogDistribution(nn.Module):

    def __init__(self, dims: int = 1, dist: str = "mse",
                 agg: str = "sum", tol: float = 1e-8) -> None:
        super().__init__()
        self.dims = dims
        self.dist = dist
        self.agg = agg
        self.tol = tol

    def forward(self, mode: Tensor) -> Distribution:
        return self.Distribution(
            mode, self.dims, self.dist,
            self.agg, self.tol)

    class Distribution:
    
        def __init__(self, mode: Tensor, dims: int = 1,
                     dist: str = "mse", agg: str = "sum",
                     tol: float = 1e-8) -> None:
            self._mode = mode
            self._dims = tuple([-x for x in range(1, dims + 1)])
            self._dist = dist
            self._agg = agg
            self._tol = tol
            self._batch_shape = mode.shape[: len(mode.shape) - dims]
            self._event_shape = mode.shape[len(mode.shape) - dims :]
    
        @property
        def mode(self) -> Tensor:
            return Utils.symexp(self._mode)
    
        @property
        def mean(self) -> Tensor:
            return Utils.symexp(self._mode)
    
        def log_prob(self, value: Tensor) -> Tensor:
            assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
            if self._dist == "mse":
                distance = (self._mode - Utils.symlog(value)) ** 2
                distance = torch.where(distance < self._tol, 0, distance)
            elif self._dist == "abs":
                distance = torch.abs(self._mode - Utils.symlog(value))
                distance = torch.where(distance < self._tol, 0, distance)
            else:
                raise NotImplementedError(self._dist)
            if self._agg == "mean":
                loss = distance.mean(self._dims)
            elif self._agg == "sum":
                loss = distance.sum(self._dims)
            else:
                raise NotImplementedError(self._agg)
            return -loss


class OneHotDistribution(nn.Module):

    def __init__(self, dims: int = 1) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, logits: Tensor) -> Tensor:
        return td.Independent(
            td.OneHotCategoricalStraightThrough(
                logits=logits), self.dims)


class MSEDistribution(nn.Module):

    def __init__(self, dims: int, agg: str = 'sum') -> None:
        super().__init__()
        self.dims = dims
        self.agg = agg

    def forward(self, mode: Tensor) -> Distribution:
        return self.Distribution(mode, self.dims, self.agg)

    class Distribution:
        def __init__(self, mode: Tensor, dims: int, agg: str = "sum") -> None:
            self._mode = mode
            self._dims = tuple([-x for x in range(1, dims + 1)])
            self._agg = agg
            self.batch_shape = mode.shape[: len(mode.shape) - dims]
            self.event_shape = mode.shape[len(mode.shape) - dims :]

        @property
        def mode(self) -> Tensor:
            return self._mode

        @property
        def mean(self) -> Tensor:
            return self._mode

        def log_prob(self, value: Tensor) -> Tensor:
            assert self._mode.shape == value.shape, (
                self._mode.shape, value.shape)
            distance = (self._mode - value) ** 2
            if self._agg == "mean":
                loss = distance.mean(self._dims)
            elif self._agg == "sum":
                loss = distance.sum(self._dims)
            else:
                raise NotImplementedError(self._agg)
            return -loss
        

class BernoulliDistribution(nn.Module):

    def __init__(self, dims: int = 1) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, logits: Tensor) -> Distribution:
        return td.Independent(self.Distribution(logits=logits), self.dims)

    class Distribution(td.Bernoulli):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        @property
        def mode(self):
            mode = (self.probs > 0.5).to(self.probs)
            return mode