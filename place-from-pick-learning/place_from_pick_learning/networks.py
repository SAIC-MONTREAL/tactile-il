"""
Basic network modules.
"""
import math
import inspect

import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

from place_from_pick_learning.utils.learning_utils import (
    MultivariateNormalDiag,
    MixtureGaussianDiag,
)

class DeterministicActionDecoder(nn.Module):
    '''
    Deterministic action output layer
    '''
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_size=256
    ):
        super().__init__()
        self.a_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        n, l = x.shape[0], x.shape[1]
        x = x.reshape(-1, *x.shape[2:])
        a = self.a_layer(x)
        a = a.reshape(n, l, *a.shape[1:])
        return (a, None)

class GaussianActionDecoder(nn.Module):
    '''
    Action output layer with Gaussian distribution as output.
    '''
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_size=256,
        use_low_noise_eval=False,
        min_std=1e-4
    ):
        super().__init__()
        self.min_std = min_std
        self.a_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * action_dim),
        )
        self.dist = MultivariateNormalDiag
        self.use_low_noise_eval = use_low_noise_eval

    def forward(self, x):
        n, l = x.shape[0], x.shape[1]
        x = x.reshape(-1, *x.shape[2:])
        a_mu, a_logvar = torch.chunk(self.a_layer(x), chunks=2, dim=-1)
        a_std = torch.exp(a_logvar / 2.0) + self.min_std
        params = (a_mu, a_std)
        params = (param.reshape(n, l, *param.shape[1:]) for param in params)
        pa_o = self.dist(*params)
        if self.use_low_noise_eval and (not self.training):
            a = a_mu
        else:
            a = pa_o.sample()
        return (a, pa_o)

class GaussianMixtureActionDecoder(nn.Module):
    '''
    Action output layer with Gaussian mixture distribution as output.
    '''
    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_size=256,
        n_mixture_components=5,
        use_low_noise_eval=False,
        min_std=1e-4
    ):
        super().__init__()

        self.mixture_components = n_mixture_components
        self.min_std = min_std
        # self.a_layer = nn.Sequential(
        #     nn.Linear(input_dim, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, self.mixture_components * 2 * action_dim)
        # )
        self.a_layer_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.mixture_components * action_dim)
        )
        self.a_layer_std = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.mixture_components * action_dim),
            nn.Softplus()
        )
        self.fc_mixture = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.mixture_components)
        )
        self.dist = MixtureGaussianDiag
        self.use_low_noise_eval = use_low_noise_eval

    def forward(self, x):
        n, l = x.shape[0], x.shape[1]
        x = x.reshape(-1, *x.shape[2:])
        # a_mu, a_logvar = torch.chunk(self.a_layer(x), chunks=2, dim=-1)
        a_mu = self.a_layer_mu(x)
        a_std = self.a_layer_std(x)

        if self.use_low_noise_eval and (not self.training):
            a_std = torch.ones_like(a_mu) * 1e-4
        else:
            # a_std = torch.exp(a_logvar / 2.0) + self.min_std
            a_std = a_std + self.min_std
        a_mu = a_mu.reshape(a_mu.shape[0], self.mixture_components, -1)
        a_std = a_std.reshape(a_std.shape[0], self.mixture_components, -1)
        a_cat = self.fc_mixture(x)
        params = (a_cat, a_mu, a_std)
        params = (param.reshape(n, l, *param.shape[1:]) for param in params)
        pa_o = self.dist(*params)
        a = pa_o.sample()
        return (a, pa_o)

class RandomShiftsAug(nn.Module):
    """Taken from: https://github.com/facebookresearch/drqv2/blob/c0c650b76c6e5d22a7eb5f2edffd1440fe94f8ef/drqv2.py#L14"""
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class MultimodalSelfAttention(nn.Module):
    """
    Self-attention across both time and modality.
    Inspired from https://openreview.net/pdf?id=sygvGP-YLfx
    """
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim (int): dimension of features of each modality
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."
        self.embed_dim = embed_dim
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
    def forward(self, x):
        """
        Forward M sequences of length L each assumed
            to have the same feature dimension of embed_dim.
        Args:
            x (list): A list of all M modalities to be fused
                where each modality has dimension (bs, L, embed_dim)
        Returns:
            out (torch.Tensor): A concatenated vector of size (bs, M * L * embed_dim)
        """
        # Concatenate M x (bs, L, embed_dim) into (bs, L * M, embed_dim)
        x = torch.cat(x, dim=1)

        # Self attention across time and modality
        # outputs (bs, L * M, embed_dim)
        out, _ = self.mha(
            query=x,
            key=x,
            value=x,
        )

        # Concatenate to (bs, M * L * embed_dim)
        out = out.view(out.shape[0], -1)
        return out

#### This block is copied from robomimic/models/base_nets.py ####
#---------------
#Visual backbone networks|
#---------------

class RNN_Base(nn.Module):
    """
    A wrapper class for a multi-step RNN and a per-step network.
    """
    def __init__(
        self,
        input_dim,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            rnn_hidden_dim (int): RNN hidden dimension
            rnn_num_layers (int): number of RNN layers
            rnn_type (str): [LSTM, GRU]
            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU
        """
        super(RNN_Base, self).__init__()

        assert rnn_type in ["LSTM", "GRU"]
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else {}
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)

        self.nets = rnn_cls(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            **rnn_kwargs,
        )

        self._hidden_dim = rnn_hidden_dim
        self._num_layers = rnn_num_layers
        self._rnn_type = rnn_type
        self._num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise

    @property
    def rnn_type(self):
        return self._rnn_type

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)
        Args:
            batch_size (int): batch size dimension
            device: device the hidden state should be sent to.
        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        h_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
        if self._rnn_type == "LSTM":
            c_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
            return h_0, c_0
        else:
            return h_0

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        out = [input_shape[0], self._num_layers * self._hidden_dim]
        return out

    def forward(self, inputs, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.
        Args:
            inputs (torch.Tensor): tensor input of shape [B, T, D], where D is the RNN input size
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state
        Returns:
            outputs: outputs of the rnn
            rnn_state: return rnn state at the end if return_state is set to True
        """
        assert inputs.ndimension() == 3  # [B, T, D]
        batch_size, seq_length, inp_dim = inputs.shape
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, device=inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_init_state)

        if return_state:
            return outputs, rnn_state
        else:
            return outputs

class ResNet18Conv(nn.Module):
    '''
    Resnet18 block. Widely used for image encoding in the literature of
    robotic manipulation using visual sensors.
    Last layer of the resnet block outputs a 512 dim vector
    '''

    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False
    ):
        '''
        input:
        :params input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
        :params pretrained (bool): if True, load pretrained weights for all ResNet layers.
        :params input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        '''
        super(ResNet18Conv, self).__init__()

        # check available arguments on resnet18, since older torch versions use a different argument
        if 'pretrained' in inspect.getfullargspec(resnet18).args:
            if pretrained:
                net = resnet18(pretrained=True)
            else:
                net = resnet18(pretrained=False)
        else:
            if pretrained:
                net = resnet18(weights='IMAGENET1K_V1')
            else:
                net = resnet18(weights=None)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

        return None

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        out_h = int(math.ceil(input_shape[1] / 32.))
        out_w = int(math.ceil(input_shape[2] / 32.))
        return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)


    def forward(self, inputs):
        x = self.nets(inputs)

        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )

        return x

class CoordConv2d(nn.Conv2d, nn.Module):
    """
    2D Coordinate Convolution
    Source: An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution
    https://arxiv.org/abs/1807.03247
    (e.g. adds 2 channels per input feature map corresponding to (x, y) location on map)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        coord_encoding='position',
    ):
        """
        Args:
            in_channels: number of channels of the input tensor [C, H, W]
            out_channels: number of output channels of the layer
            kernel_size: convolution kernel size
            stride: conv stride
            padding: conv padding
            dilation: conv dilation
            groups: conv groups
            bias: conv bias
            padding_mode: conv padding mode
            coord_encoding: type of coordinate encoding. currently only 'position' is implemented
        """

        assert(coord_encoding in ['position'])
        self.coord_encoding = coord_encoding
        if coord_encoding == 'position':
            in_channels += 2  # two extra channel for positional encoding
            self._position_enc = None  # position encoding
        else:
            raise Exception("CoordConv2d: coord encoding {} not implemented".format(self.coord_encoding))
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, input):
        b, c, h, w = input.shape
        if self.coord_encoding == 'position':
            if self._position_enc is None:
                pos_y, pos_x = torch.meshgrid(
                    torch.arange(h),
                    torch.arange(w),
                    indexing='ij'
                )
                pos_y = pos_y.float().to(input.device) / float(h)
                pos_x = pos_x.float().to(input.device) / float(w)
                self._position_enc = torch.stack((pos_y, pos_x)).unsqueeze(0)
            pos_enc = self._position_enc.expand(b, -1, -1, -1)
            input = torch.cat((input, pos_enc), dim=1)
        return super(CoordConv2d, self).forward(input)

####  ################################################ #####

class SpatialSoftArgmax(nn.Module):
    """Spatial softmax as defined in [1].
    Concretely, the spatial softmax of each feature
    map is used to compute a weighted mean of the pixel
    locations, effectively performing a soft arg-max
    over the feature dimension.
    References:
        [1]: End-to-End Training of Deep Visuomotor Policies,
        https://arxiv.org/abs/1504.00702
    """

    def __init__(self, normalize=True):
        """Constructor.
        Args:
            normalize (bool): Whether to use normalized
                image coordinates, i.e. coordinates in
                the range `[-1, 1]`.
        """
        super().__init__()

        self.normalize = normalize
        #TODO: CNN layer to downsample and control KPs

    def _coord_grid(self, h, w, device):
        if self.normalize:
            return torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, w, device=device),
                    torch.linspace(-1, 1, h, device=device),
                    indexing='ij'
                )
            )
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, w, device=device),
                torch.arange(0, h, device=device),
                indexing='ij'
            )
        )

    def forward(self, x):
        assert x.ndim == 4, "Expecting a tensor of shape (B, C, H, W)."
        b, c, h, w = x.shape

        softmax = nn.functional.softmax(x.reshape(-1, h * w), dim=-1)

        xc, yc = self._coord_grid(h, w, x.device)
        x_mean = (softmax * xc.flatten()).sum(dim=1, keepdims=True)
        y_mean = (softmax * yc.flatten()).sum(dim=1, keepdims=True)

        return torch.cat([x_mean, y_mean], dim=1).view(-1, c * 2)
