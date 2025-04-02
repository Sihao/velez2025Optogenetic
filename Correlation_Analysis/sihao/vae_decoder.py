import torch
import torch.nn as nn
from typing import List
from typing import Union
from typing import TypeVar
import torch.nn.functional as F

Self = TypeVar('Self', bound = 'Decoder')

class Decoder(nn.Module):
    """
    Decoder of the Variational AutoEncoder
        This class implements the convolutional decoder of the VAE. 
        Initiation requires: 
            1. cfg = The list of network layers that was passed to the encoder 
            2. out_features = The number of timepoints to reconstuct 
            3. latent_dim = The number of latent dimensions
            4. max_pool_kernel = The kernel size for the max pool layers 
    """

    def __init__(self: Self, 
                 cfg: List[Union[int, str]], # list of network layers 
                 out_features: int, # number of features to reconstruct (e.g. 3400)
                 latent_dim: int, # number of latent dimensions (e.g. 16)
                 max_pool_kernel: int # size of max pool kernel 
                 ) -> Self:
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.max_pool_kernel = max_pool_kernel
        # Calculate the number of downsampled features after
        # max pool layers and convolutional layers 
        self.downsampled_features = out_features//max_pool_kernel**sum([x=='M' for x in cfg])
        # Use that feature size to determine the size of the first linear layer 
        self.fc_1 = nn.Linear(latent_dim, self.downsampled_features * cfg[0])         
        # Generate network of layers 
        self.features = self.make_layers(cfg[1:], cfg[0]) 
        # Initialize weights 
        self._initialize_weights()
        # Get rid of last activation function and replace with tanh 
        self.features[-1] = nn.Tanh()

    def forward(self: Self, x: torch.Tensor):
        """
        Decoder the input by passing through the decoder network
        and returns the reconstruction x_hat 
        :param input: (Tensor) Input tensor to decoder [B x F]
        :return: (Tensor) List of latent codes
        """
        x = self.fc_1(x)
        # resize output of first fully connected linear layer into 
        # output channels and downsampled features 
        x = x.reshape(x.size(0), self.cfg[0], self.downsampled_features)
        out = self.features(x) # series of convolutional layers 
#        out = F.tanh(out) # end with inverse tangent bc output is between -1 and 1 
        return out

    def _initialize_weights(self: Self) -> None:
        """
        Initializes weights in the network using Kaiming Uniform 
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='leaky_relu')
                print(f'Decoder Weights of Shape: {m.weight.size()}')

    # Create layer architecture of encoder 
    def make_layers(self: Self,
                    cfg: List[Union[int, str]],
                    in_channels: int, 
                    ) -> nn.Sequential:
        """
        Creates layer architecture of decoder
        Each 1D deconvolution is followed by batch normalization 
        and a leaky rectified linear unit activation function 
        The kernel size of the max pool layers is determined 
        by the global self.max_pool_kernel size 
        """
        layers: List[nn.Module] = []
        print(f'Making layers for Decoder: {cfg}')
        for v in cfg:
            if v == 'M':
                layers.append(nn.Upsample(scale_factor = self.max_pool_kernel, mode = 'nearest'))
            else:
                layers.append(nn.ConvTranspose1d(in_channels, v, kernel_size = 3, stride = 1, padding = 1))
                layers.append(nn.BatchNorm1d(v))
                layers.append(nn.LeakyReLU())
                in_channels = v
        return nn.Sequential(*layers)
    