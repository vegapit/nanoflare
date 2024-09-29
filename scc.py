import torch
from torch import nn
from collections import OrderedDict
from audiomodel import AudioModel, ConvClipper

class SCC( AudioModel ):
    def __init__(self, kernel_size, depth_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.kernel_size = kernel_size
        self.depth_size = depth_size
        self.stack = nn.Sequential(
            OrderedDict([(f"{i}", ConvClipper( kernel_size, 2**i )) for i in range(depth_size)])
        )
    def forward(self, x):
        return self.stack( x )
    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'SCC',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'kernel_size': self.kernel_size,
                'depth_size': self.depth_size
            },
            'state_dict': { f'stack.{i}': unit.generate_doc() for i, unit in enumerate(self.stack) }
        }
        return doc
if __name__ == "__main__":
    model = SCC( 128, 3 )

    model.eval()

    x = torch.randn((1, 1024)).to(torch.float32)
    y = model( x )
    
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )