import torch
import torch.nn as nn
from typing import List
from typing import Union
from typing import TypeVar

Self = TypeVar('Self', bound = 'Encoder')

# Ultimately I need to split the Variational part from the main Encoder 
# Forward() is called automatically by lightning 

class Encoder(nn.Module):
    """
    Encoder of the Variational AutoEncoder
        This class implements the convolutional encoder of the VAE. 
        Initiation requires: 
            1. cfg = A list of network layers
            2. in_features = The number of timepoints
            3. latent_dim = The number of latent dimensions
            4. max_pool_kernel = The kernel size for the max pool layers 
    """
    def __init__(self: Self, 
                 cfg: List[Union[int, str]],
                 in_features: int, 
                 in_channels: int, 
                 latent_dim: int, 
                 max_pool_kernel: int 
                 ) -> Self:
        super(Encoder, self).__init__()
        self.max_pool_kernel = max_pool_kernel
        self.features = self.make_layers(cfg, in_channels) # Make convolutional layers 
        # Calculate down-sampled features after max-pooling and 
        # multiply by final number of channels
        downsampled_features = (in_features//max_pool_kernel**sum([x=='M' for x in cfg])) * cfg[-1]
        # Use down-sampled features to convert to latent dimensions 
        self.fc_mu = nn.Linear(downsampled_features, latent_dim) 
        self.fc_logvar = nn.Linear(downsampled_features, latent_dim)
        # initialize weights in network 
        self._initialize_weights()
        
    def forward(self: Self, x: torch.Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes mu and logvar 
        :param input: (Tensor) Input tensor to encoder [B x C x F]
        :return: (Tensor) List of latent codes
        """
        out = self.features(x) # series of convolutional layers  
        out = torch.flatten(out, start_dim=1) # combine feature and channel dimensions 
        mu = self.fc_mu(out) 
        logvar = self.fc_logvar(out)
        return mu, logvar

    def _initialize_weights(self: Self) -> None:
        """
        Initializes weights in the network using Kaiming Uniform 
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='leaky_relu')
                print(f'Encoder Weights of Shape: {m.weight.size()}')

    # Create layer architecture of encoder 
    def make_layers(self: Self,
                    cfg: List[Union[int, str]],
                    in_channels: int, 
                    ) -> nn.Sequential:
        """
        Creates layer architecture of encoder
        Each 1D convolution is followed by batch normalization 
        and a leaky rectified linear unit activation function 
        The kernel size of the max pool layers is determined 
        by the global self.max_pool_kernel size 
        """
        layers: List[nn.Module] = []
        print(f'Making layers for Encoder: {cfg}')
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = self.max_pool_kernel, padding = 0))
            else:
                layers.append(nn.Conv1d(in_channels, v, kernel_size = 3, stride = 1, padding = 1))
                layers.append(nn.BatchNorm1d(v))
                layers.append(nn.LeakyReLU())
                in_channels = v 
        return nn.Sequential(*layers)
 

