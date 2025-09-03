# Imports
import copy
import torch
from torch import Tensor, nn
import numpy as np
from tensordict import TensorDict
from distributions import *
from utils import Config, Utils
import operator

#####################################################################
#   General Neural Network Modules
#####################################################################

class MLP(nn.Module):

    def __init__(self, in_dim: int, n_layers: int, 
                 hx_dim: int, out_dim: int, HxActivation: nn.Module,
                 OutActivation: nn.Module=None, out_norm: bool=False,
                 out_dist: nn.Module=None, preprocessing: nn.Module=None,
                 postprocessing: nn.Module=None) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.Sequential()
        if preprocessing: self.layers.add_module('preprocessing', preprocessing)
        final_out_dim = out_dim
        if n_layers > 1: out_dim = hx_dim 
        for i in range(1, n_layers + 1):
            if i < n_layers:
                self.layers.add_module(f'linear{i}', nn.Linear(in_dim, out_dim, False))
                self.layers.add_module(f'norm{i}', nn.LayerNorm(out_dim))
                self.layers.add_module(f'act{i}', HxActivation())
                in_dim = out_dim
            elif out_norm:
                self.layers.add_module(f'linear{i}', nn.Linear(in_dim, out_dim, False))
                self.layers.add_module(f'norm{i}', nn.LayerNorm(out_dim))
                if OutActivation: self.layers.add_module(f'act{i}', OutActivation())
                if postprocessing: self.layers.add_module('postprocessing', postprocessing)
                if out_dist: self.layers.add_module(f'out_dist', out_dist)
            else:
                self.layers.add_module(f'linear{i}', nn.Linear(in_dim, final_out_dim))
                if OutActivation: self.layers.add_module(f'act{i}', OutActivation())
                if postprocessing: self.layers.add_module('postprocessing', postprocessing)
                if out_dist: self.layers.add_module(f'out_dist', out_dist)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        return y

class ChannelLayerNorm(nn.LayerNorm):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, -1)
        y = super().forward(x)
        y = y.transpose(1, -1)
        return y
    
class Encoder1D(MLP):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Decoder1D(MLP):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class Encoder2D(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int,
                 Activation: nn.Module, n_layers: int=4,
                 residual: bool=False, **kwargs) -> None:
        super().__init__()
        self.n_layers = n_layers
        out_dim = hidden_dim
        self.layers = nn.Sequential()
        for i in range(n_layers):
            if residual:
                self.layers.add_module(f'resblk{i+1}', self.ResidualBlock(in_dim, Activation, ChannelLayerNorm, True, out_dim))
            else:
                self.layers.add_module(f'conv{i+1}', nn.Conv2d(in_dim, out_dim, 4, 2, 1, bias=False))
                self.layers.add_module(f'norm{i+1}', ChannelLayerNorm(out_dim))
                self.layers.add_module(f'act{i+1}', Activation())
            in_dim = 2**i * hidden_dim 
            out_dim = 2**(i+1) * hidden_dim 
        self.layers.add_module(f'flatten1', nn.Flatten())

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        return y

    class ResidualBlock(nn.Module):

        def __init__(self, in_channels: int, Act: nn.Module,
                     Norm: nn.Module, downsample: bool=False,
                     width: int|None=None) -> None:
            super().__init__()
            self.downsample = downsample
            self.layers = nn.Sequential()
            self.final_act = Act()
            if downsample:
                assert width is not None, 'width must be defined for downsample residual block!'
                self.layers.add_module('conv1', nn.Conv2d(in_channels, width, 3, 2, 1, bias=False))
                self.layers.add_module('norm1', Norm(width))
                self.layers.add_module('act1', Act())
                self.layers.add_module('conv2', nn.Conv2d(width, width, 3, 1, 1, bias=False, groups=width))
                self.layers.add_module('norm2', Norm(width))
                self.downsample_layers = nn.Sequential()
                self.downsample_layers.add_module('downsample_conv', nn.Conv2d(in_channels, width, 1, 2, bias=False))
                self.downsample_layers.add_module('downsample_norm', Norm(width))
            else:
                self.layers.add_module('conv1', nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
                self.layers.add_module('norm1', Norm(in_channels))
                self.layers.add_module('act1', Act())
                self.layers.add_module('conv2', nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False))
                self.layers.add_module('norm2', Norm(in_channels))

        def forward(self, x: Tensor) -> Tensor:
            skip = x
            x = self.layers(x)
            if self.downsample:
                skip = self.downsample_layers(skip)
            y = self.final_act(x + skip)
            return y

class Decoder2D(nn.Module):
    
    def __init__(self, in_dim: int, out_dim: int,
                 hidden_dim: int, HxActivation: nn.Module,
                 OutActivation: nn.Module, output_shape: tuple[int],
                 n_layers: int=4, out_dist: Distribution=None, residual: bool=False) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.Sequential()
        final_out_dim = out_dim
        out_h, out_w = output_shape
        in_featmap_width = out_w//16
        in_featmap_height = out_h//16
        out_dim = 2**(n_layers-1) * hidden_dim * in_featmap_width * in_featmap_height
        self.layers.add_module('linear1', MLP(in_dim, 1, None, out_dim,
                                              None, HxActivation, True))
        self.layers.add_module('unflatten1', nn.Unflatten(-1, (-1, in_featmap_height, in_featmap_width)))
        for i in range(n_layers):
            if i+1 < n_layers:
                in_dim = 2**(n_layers - i - 1) * hidden_dim 
                out_dim = 2**(n_layers - i - 2) * hidden_dim 
                if residual:
                    self.layers.add_module(f'resblk{i+1}', self.ResidualBlock(
                        in_dim, HxActivation, ChannelLayerNorm, True, out_dim))
                else:
                    self.layers.add_module(f'tconv{i+1}', nn.ConvTranspose2d(
                        in_dim, out_dim, 4, 2, 1, bias=False))
                    self.layers.add_module(f'norm{i+1}', ChannelLayerNorm(out_dim))
                    self.layers.add_module(f'act{i+1}', HxActivation())
            else:
                in_dim = out_dim
                self.layers.add_module(f'tconv{i+1}', nn.ConvTranspose2d(
                    in_dim, final_out_dim, 4, 2, 1, bias=True))
                if OutActivation: self.layers.add_module(f'act{i+1}', OutActivation())
                if out_dist: self.layers.add_module(f'out_dist', out_dist)

    def forward(self, x: Tensor) -> Tensor:
        y = self.layers(x)
        return y

    class ResidualBlock(nn.Module):

        def __init__(self, in_channels: int, Act: nn.Module,
                     Norm: nn.Module, upsample: bool=False,
                     width: int|None=None) -> None:
            super().__init__()
            self.upsample = upsample
            self.layers = nn.Sequential()
            self.final_act = Act()
            if upsample:
                assert width is not None, 'width must be defined for downsample residual block!'
                self.layers.add_module('tconv1', nn.ConvTranspose2d(in_channels, width, 4, 2, 1, bias=False))
                self.layers.add_module('norm1', Norm(width))
                self.layers.add_module('act1', Act())
                self.layers.add_module('tconv2', nn.ConvTranspose2d(width, width, 3, 1, 1, bias=False, groups=width))
                self.layers.add_module('norm2', Norm(width))
                self.upsample_layers = nn.Sequential()
                self.upsample_layers.add_module('upsample_conv', nn.ConvTranspose2d(in_channels, width, 2, 2, bias=False))
                self.upsample_layers.add_module('upsample_norm', Norm(width))
            else:
                self.layers.add_module('tconv1', nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False))
                self.layers.add_module('norm1', Norm(in_channels))
                self.layers.add_module('act1', Act())
                self.layers.add_module('tconv2', nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1, bias=False))
                self.layers.add_module('norm2', Norm(in_channels))

        def forward(self, x: Tensor) -> Tensor:
            skip = x
            x = self.layers(x)
            if self.upsample:
                skip = self.upsample_layers(skip)
            y = self.final_act(x + skip)
            return y

class RecurrentNet(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int,
                    recurrent_dim: int, Activation: nn.Module) -> None:
        super().__init__()
        self.mlp = MLP(input_dim, 1, None, hidden_dim,
                       None, Activation, True)
        self.gru = self.LayerNormGRU(hidden_dim, recurrent_dim)

    def forward(self, x: Tensor, hx: Tensor) -> Tensor:
        h = self.mlp(x)
        hx = self.gru(h, hx)
        return hx
    
    class LayerNormGRU(nn.Module):

        def __init__(self, input_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.linear = nn.Linear(input_dim + hidden_dim,
                                    3 * hidden_dim, bias=False)
            self.norm = nn.LayerNorm(3 * hidden_dim)

        def forward(self, x: Tensor, hx: Tensor) -> Tensor:
            x = self.norm(self.linear(torch.cat((x, hx), -1)))
            reset, new, upd = x.chunk(3, -1)
            reset_gate = torch.sigmoid(reset)
            new_gate = torch.tanh(reset_gate * new)
            update_gate = torch.sigmoid(upd - 1)
            hx = update_gate * new_gate + (1 - update_gate) * hx
            return hx

#####################################################################
#   Specific Neural Networks
#####################################################################

class WorldModel(nn.Module):

    def __init__(self, encoders: Config, decoders: Config,
                 action_dim: int, Act: nn.Module, cnn_dim: int,
                 hidden_dim: int, recurrent_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_shape = ((latent_dim),)*2
        # Encoder Networks: maps obs_t -> enc_t
        self.encoder_names = []; encoded_dim = 0
        for name, spec in encoders.to_dict().items():
            network_type = next(iter(spec.to_dict().keys()))
            encoder_args = {}
            if network_type == 'networks.Encoder2D':
                input_h, input_w = next(iter(spec.to_dict().values())).input_shape
                encoder_args['hidden_dim'] = cnn_dim
                encoder_args['Activation'] = Act
                encoded_dim += (2**3) * cnn_dim * input_h//16 * input_w//16
            elif network_type == 'networks.Encoder1D':
                encoder_args['n_layers'] = 3
                encoder_args['hx_dim'] = hidden_dim
                encoder_args['out_dim'] = hidden_dim 
                encoder_args['HxActivation'] = Act
                encoder_args['out_norm'] = True
                encoder_args['preprocessing'] = Utils.Symlog()
                encoded_dim += encoder_args['out_dim']
            else:
                raise NotImplementedError(network_type)
            setattr(self, f'{name}_encoder', spec.create_instance(encoder_args))
            self.encoder_names += [name]
        # Representation Network: maps enc_t, h_t -> z_t
        self.representation_net = MLP(
            encoded_dim + recurrent_dim, 2, hidden_dim, latent_dim**2, Act,
            postprocessing=nn.Sequential(nn.Unflatten(-1, self.latent_shape), Utils.UniformMix()),
            out_dist=OneHotDistribution())
        # Sequence Network: maps (z_t, a_t, h_t) -> h_{t+1}
        self.h0 = nn.Parameter(torch.zeros(1, recurrent_dim))
        self.a0 = nn.Parameter(torch.zeros(1, action_dim))
        self.sequence_net = RecurrentNet(latent_dim**2 + action_dim,
                                         hidden_dim, recurrent_dim, Act)
        # Dynamics Network: maps h_t -> ẑ_t
        self.dynamics_net = MLP(
            recurrent_dim, 2, hidden_dim, latent_dim**2, Act,
            postprocessing=nn.Sequential(nn.Unflatten(-1, self.latent_shape), Utils.UniformMix()),
            out_dist=OneHotDistribution())
        # Decoder Networks: maps s_t -> obs_t 
        self.decoder_names = []
        for name, spec in decoders.to_dict().items():
            network_type = next(iter(spec.to_dict().keys()))
            decoder_args = {}
            if network_type == 'networks.Decoder2D':
                decoder_args['in_dim'] = latent_dim**2 + recurrent_dim
                decoder_args['hidden_dim'] = cnn_dim
                decoder_args['HxActivation'] = Act
                decoder_args['OutActivation'] = nn.Sigmoid
                decoder_args['out_dist'] = MSEDistribution(dims=3)
            elif network_type == 'networks.Decoder1D':
                decoder_args['in_dim'] = latent_dim**2 + recurrent_dim
                decoder_args['n_layers'] = 4
                decoder_args['hx_dim'] = hidden_dim
                decoder_args['HxActivation'] = Act
                decoder_args['out_dist'] = SymlogDistribution()
            else:
                raise NotImplementedError(network_type)
            setattr(self, f'{name}_decoder', spec.create_instance(decoder_args))
            self.decoder_names += [name]
        # Reward Network: maps s_t -> r_t 
        self.reward_net = MLP(recurrent_dim + latent_dim**2, 2, hidden_dim,
                              255, Act, out_dist=TwoHotDistribution())
        # Continue Network: maps s_t -> c_t
        self.continue_net = MLP(recurrent_dim + latent_dim**2, 2, hidden_dim,
                                1, Act, out_dist=BernoulliDistribution())

    def _forward_encoders(self, obs: TensorDict) -> Tensor:
        encoded = torch.cat([getattr(self, f'{name}_encoder')(
            obs[name].flatten(0,1)).unflatten(
                0, obs.batch_size) for name in self.encoder_names], -1)
        return encoded

    def _forward_decoders(self, states: Tensor) -> TensorDict:
        decoded = {}
        b,t = states.shape[:2]
        states = states.flatten(0,1)
        for name in self.decoder_names:
            d = getattr(self, f'{name}_decoder')(states)
            decoded[name] = type(d)(mode=d.mode.unflatten(0, (b,t)), dims=len(d._dims), agg=d._agg)
        return TensorDict(decoded, [b,t])
 
    def forward(self, obs: TensorDict, actions: Tensor, continues: Tensor) -> tuple[Tensor, Distribution]:
        # Get initial recurrent state
        h_t = self._get_inital_recurrent_state(actions.size(0))
        # Encode observations
        enc = self._forward_encoders(obs)
        # Initialize lists for storing step results
        (posterior_logits, states,
         posteriors, recurrents) = [], [], [], []
        # For each timestep...
        for enc_t, a_t, c_t in zip(enc.transpose(0,1), actions.transpose(0,1), continues.transpose(0,1)):
            # Calculate the posterior distribution logits
            posterior_distribution_t = self.representation_net(torch.cat((enc_t, h_t), -1))
            z_t = posterior_distribution_t.rsample()
            # Form the current state s_t
            s_t = torch.cat((h_t, z_t.flatten(-2)), -1)
            # Record step calculations
            states += [s_t]
            posteriors += [z_t]
            posterior_logits += [posterior_distribution_t.base_dist.logits]
            recurrents += [h_t]
            # Calculate the next recurrent hidden state h_t+1
            za_t = torch.cat((z_t.flatten(-2), a_t), -1)
            # Calculate new recurrent vector (reset if done)
            c_t = c_t.squeeze(-1)
            h_t[c_t] = self.sequence_net(za_t[c_t], h_t[c_t])
            h_t[~c_t] = self._get_inital_recurrent_state((~c_t).sum())
        # Combine sequences/create posterior distribution
        posterior_distribution = Utils.one_hot_cat_dist(
            torch.stack(posterior_logits).transpose(0,1))
        posteriors = torch.stack(posteriors).transpose(0,1)
        recurrents = torch.stack(recurrents).transpose(0,1)
        states = torch.stack(states).transpose(0,1)
        # Calculate the prior distribution
        prior_distribution = self.dynamics_net(recurrents)
        # Calculate the reward distribution
        reward_distribution = self.reward_net(states)
        # Calculate the continue distribution
        continue_distribution = self.continue_net(states)
        # Estimate the observation x̂ from the state
        est_observation_distribution = self._forward_decoders(states)
        return (prior_distribution, posterior_distribution, est_observation_distribution, 
                reward_distribution, continue_distribution, posteriors, recurrents)

    def dream(self, initial_posterior: Tensor, initial_recurrent: Tensor,
              actor: nn.Module, n_steps: int, exploit: bool = False) -> tuple[Tensor, Distribution]:
       # Initialize lists for storing step results
        states, actions, action_stats = [], [], []
        # set initial states as current states
        h_t = initial_recurrent; z_t = initial_posterior
        # for N timesteps...
        for _ in range(n_steps + 1):
            # Form the current state
            z_t = z_t.flatten(-2)
            s_t = torch.cat((h_t, z_t), -1)
            # Calculate action from the actor policy
            action_distribution_t = actor(s_t.detach())
            if exploit: a_t = action_distribution_t.mode
            else: a_t = action_distribution_t.rsample()
            za_t = torch.cat((z_t, a_t), -1)
            # Calculate the next recurrent vector
            h_t = self.sequence_net(za_t, h_t)
            # Calculate the current *prior* latent state
            prior_distribution_t = self.dynamics_net(h_t)
            z_t = prior_distribution_t.rsample()
            states += [s_t]; actions += [a_t]
            action_stats += [torch.cat((action_distribution_t.loc,
                                        action_distribution_t.scale), -1)]
        # Combine step results to form sequence tensors
        states = torch.stack(states).transpose(0,1)
        actions = torch.stack(actions).transpose(0,1)
        action_stats = torch.stack(action_stats).transpose(0,1)
        action_loc, action_scale = action_stats.chunk(2, -1)
        action_distribution = actor.layers.out_dist(loc=action_loc, scale=action_scale)
        continue_distribution = self.continue_net(states)
        continues = continue_distribution.mode
        # Compute reward distribution
        reward_distribution = self.reward_net(states)
        return (states, actions, continues, action_distribution,
                reward_distribution, continue_distribution)
    
    def encode_state(self, obs_t: TensorDict, h_t: Tensor, exploit: bool = False) -> tuple[Tensor]:
        enc_t = self._forward_encoders(obs_t.unsqueeze(0)).squeeze(0)
        posterior_distribution = self.representation_net(torch.cat((enc_t, h_t), -1))
        if exploit: z_t = posterior_distribution.mode.flatten(-2)
        else: z_t = posterior_distribution.rsample().flatten(-2)
        s_t = torch.cat((h_t, z_t), -1)
        return s_t, z_t

    def _get_inital_recurrent_state(self, batch_size: int) -> Tensor:
        h_t = self.h0.expand(batch_size, -1).tanh()
        prior_distribution = self.dynamics_net(h_t)
        z_t = prior_distribution.mode.flatten(-2)
        a_t = self.a0.expand(batch_size, -1).tanh()
        za_t = torch.cat((z_t, a_t), -1)
        h_t = self.sequence_net(za_t, h_t)
        return h_t

class Dreamer(nn.Module):

    NET_CFG = {
        'Activation': nn.SiLU,
        'smol' : {'hidden_dim': 64,  'recurrent_dim': 64,  'cnn_dim': 4, 'latent_dim': 4},
        'XS' : {'hidden_dim': 256,  'recurrent_dim': 256,  'cnn_dim': 16, 'latent_dim': 16},
        'S'  : {'hidden_dim': 512,  'recurrent_dim': 512,  'cnn_dim': 24, 'latent_dim': 24},
        'M'  : {'hidden_dim': 640,  'recurrent_dim': 1024, 'cnn_dim': 32, 'latent_dim': 32},
        'L'  : {'hidden_dim': 768,  'recurrent_dim': 2048, 'cnn_dim': 48, 'latent_dim': 48},
        'XL' : {'hidden_dim': 1024, 'recurrent_dim': 4096, 'cnn_dim': 64, 'latent_dim': 64},
        'XXL': {'hidden_dim': 1536, 'recurrent_dim': 6144, 'cnn_dim': 96, 'latent_dim': 96},
    }

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        net_cfg = Config(**self.NET_CFG[cfg.size])
        self.world_model = cfg.world_model.create_instance(
            dict(Act=self.NET_CFG['Activation'],
                 action_dim=cfg.action_dim,
                 cnn_dim=net_cfg.cnn_dim,
                 hidden_dim=net_cfg.hidden_dim,
                 recurrent_dim=net_cfg.recurrent_dim,
                 latent_dim=net_cfg.latent_dim))
        self.actor = cfg.actor.create_instance(
            dict(in_dim=net_cfg.recurrent_dim + self.world_model.latent_dim**2,
                 n_layers=4,
                 hx_dim=net_cfg.hidden_dim,
                 HxActivation=self.NET_CFG['Activation'],
                 out_dim=cfg.action_dim * 2,
                 out_dist=TruncNormalDistribution()))
        self.critic = cfg.actor.create_instance(
            dict(in_dim=net_cfg.recurrent_dim + self.world_model.latent_dim**2,
                 n_layers=4,
                 hx_dim=net_cfg.hidden_dim,
                 HxActivation=self.NET_CFG['Activation'],
                 out_dim=255, out_dist=TwoHotDistribution()))
        self.target_critic = copy.deepcopy(self.critic)
        self._initialize_parameters()

    def update_target_critic(self, tau: float) -> None:
        for cp, tcp in zip(self.critic.parameters(),
                           self.target_critic.parameters()):
            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

    def _initialize_parameters(self) -> None:
        # Adapted from: https://github.com/NM512/dreamerv3-torch/blob/main/tools.py#L957
        def init_weights(m):
            if isinstance(m, nn.Linear):
                in_num = m.in_features
                out_num = m.out_features
                denoms = (in_num + out_num) / 2.0
                scale = 1.0 / denoms
                std = np.sqrt(scale) / 0.87962566103423978
                nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                space = m.kernel_size[0] * m.kernel_size[1]
                in_num = space * m.in_channels
                out_num = space * m.out_channels
                denoms = (in_num + out_num) / 2.0
                scale = 1.0 / denoms
                std = np.sqrt(scale) / 0.87962566103423978
                nn.init.trunc_normal_(m.weight.data, mean=0.0, std=std, a=-2.0, b=2.0)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1.0)
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        def uniform_init_weights(given_scale):
            def f(m):
                if isinstance(m, nn.Linear):
                    in_num = m.in_features
                    out_num = m.out_features
                    denoms = (in_num + out_num) / 2.0
                    scale = given_scale / denoms
                    limit = np.sqrt(3 * scale)
                    nn.init.uniform_(m.weight.data, a=-limit, b=limit)
                    if hasattr(m.bias, "data"):
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.LayerNorm):
                    m.weight.data.fill_(1.0)
                    if hasattr(m.bias, "data"):
                        m.bias.data.fill_(0.0)
            return f

        self.apply(init_weights)
        last_layer_idx = self.world_model.reward_net.n_layers
        getattr(self.world_model.reward_net.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(0.0))
        # Dynamics-Network
        last_layer_idx = self.world_model.dynamics_net.n_layers
        getattr(self.world_model.dynamics_net.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(1.0))
        # Decoder-Network
        for name in self.world_model.decoder_names:
            net = getattr(self.world_model, f'{name}_decoder')
            last_layer_idx = net.n_layers
            if isinstance(net, Decoder2D):
                getattr(net.layers, f'tconv{last_layer_idx}').apply(uniform_init_weights(1.0))
            elif isinstance(net, MLP):
                getattr(net.layers, f'linear{last_layer_idx}').apply(uniform_init_weights(1.0))
        # Representation-Network
        last_layer_idx = self.world_model.representation_net.n_layers
        getattr(self.world_model.representation_net.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(1.0))
        # Continue-Network
        last_layer_idx = self.world_model.continue_net.n_layers
        getattr(self.world_model.continue_net.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(1.0))
        # Actor-Network
        last_layer_idx = self.actor.n_layers
        getattr(self.actor.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(1.0))
        self.target_critic = copy.deepcopy(self.critic)
        # Critic-Network
        last_layer_idx = self.critic.n_layers
        getattr(self.critic.layers,
                f'linear{last_layer_idx}').apply(uniform_init_weights(0.0))
        self.target_critic = copy.deepcopy(self.critic)

    def deployable(self) -> nn.Module:
        return self.DeployableDreamer(self)

    class DeployableDreamer(nn.Module):
        # TODO
        def __init__(self, dreamer: nn.Module) -> None:
            super().__init__()
            # Encoder Networks: maps obs_t -> enc_t
            self.encoder_names = dreamer.world_model.encoder_names
            for name in self.encoder_names:
                setattr(self, f'{name}_encoder', getattr(dreamer.world_model, f'{name}_encoder'))
            # Representation Network: maps enc_t, h_t -> z_t
            self.representation_net = dreamer.world_model.representation_net
            # Sequence Network: maps (z_t, a_t, h_t) -> h_{t+1}
            self.sequence_net = dreamer.world_model.sequence_net
            # Form State (z_t, h_{t+1}) -> s_t
            # Actor Network: maps s_t -> a_{t+1}
            self.actor_net = dreamer.actor
            # replace layernorms
            #for name, layer in self.named_modules():
            #    if isinstance(layer, (nn.LayerNorm, ChannelLayerNorm)):
            #        name = name.split('.')
            #        l = operator.attrgetter('.'.join(name[:-1]))(self)
            #        setattr(l, name[-1], nn.Identity())

        def forward(self, obs_t: tuple[Tensor], h_t: Tensor) -> Tensor:
            enc_t = torch.cat([getattr(self, f'{name}_encoder')(obs) for name, obs in zip(self.encoder_names, obs_t)], -1)
            posterior_distribution = self.representation_net(torch.cat((enc_t, h_t), -1))
            #z_t = posterior_distribution.mode.flatten(-2)
            z_t = posterior_distribution.sample().flatten(-2)
            s_t = torch.cat((h_t, z_t), -1)
            action_distribution = self.actor_net(s_t)
            a_t = action_distribution.mode
            za_t = torch.cat((z_t, a_t), -1)
            h_t = self.sequence_net(za_t, h_t)
            return a_t, h_t
