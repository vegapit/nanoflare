import torch
from torch import nn
from collections import OrderedDict
from .modules import BaseModel, ConvClipper

class ConvWaveshaper( BaseModel ):
    def __init__(self, kernel_size, depth_size, num_channels, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.kernel_size = kernel_size
        self.depth_size = depth_size
        self.num_channels = num_channels
        self.stack = nn.Sequential(
            OrderedDict(
                [("0", ConvClipper( 1, num_channels, kernel_size, 1 ))]
                + [(f"{i}", ConvClipper( num_channels, num_channels, kernel_size, 2**i )) for i in range(1, depth_size - 1)]
                + [(f"{depth_size-1}", ConvClipper( num_channels, 1, kernel_size, 2**(depth_size-1)))]
            )
        )
    def forward(self, x):
        x = self.normalise( x )
        return self.stack( x )
    def generate_doc(self, meta_data={}):
        doc = {
            'config': {
                'model_type': 'ConvWaveshaper',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'meta_data': meta_data,
            'parameters': {
                'kernel_size': self.kernel_size,
                'depth_size': self.depth_size,
                'num_channels': self.num_channels
            },
            'state_dict': { f'stack.{i}': unit.generate_doc() for i, unit in enumerate(self.stack) }
        }
        return doc

if __name__ == "__main__":
    model = ConvWaveshaper( 128, 3, 8 )

    model.eval()

    x = torch.randn((1, 1024)).to(torch.float32)
    y = model( x )
    
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )