import torch
import torch.nn as nn
from .modules import BaseModel, CausalDilatedConv1d, ResidualBlock

class WaveNet( BaseModel ):
    def __init__(self, input_size, num_channels, output_size, kernel_size, dilations, stack_size, gated, hidden_size, norm_mean=0.0, norm_std=1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.num_channels = num_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.stack_size = stack_size
        self.gated = gated
        self.hidden_size = hidden_size

        self.input_conv = CausalDilatedConv1d(input_size, num_channels, kernel_size, 1)
        self.block_stack = nn.ModuleList([
            ResidualBlock(num_channels, kernel_size, dilations[i % len(dilations)], gated) 
            for i in range(stack_size * len(dilations))
        ])

        self.post_conv1 = nn.Conv1d(num_channels, hidden_size, 1)
        self.relu = nn.ReLU(inplace=True)
        self.post_conv2 = nn.Conv1d(hidden_size, output_size, 1)

        self.skip_scale = 1.0 / (stack_size * len(dilations))**0.5

    def forward(self, x):
        x = self.normalise(x)
        y = self.input_conv( x )
        skip_connections = []
        for block in self.block_stack:
            y, skip_y = block( y )
            skip_connections.append( skip_y )
        
        skip_sum = torch.stack(skip_connections, dim=0).sum(dim=0) * self.skip_scale
        
        # Post-processing
        out = self.relu(skip_sum)
        out = self.post_conv1(out)
        out = self.relu(out)
        out = self.post_conv2(out)

        return self.denormalise(out)

    def generate_doc(self, meta_data={}):
        doc = {
            'config': {
                'model_type': 'WaveNet',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'meta_data': meta_data,
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dilations': self.dilations,
                'stack_size': self.stack_size,
                'gated': self.gated,
                'hidden_size': self.hidden_size,
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'input_conv': self.input_conv.generate_doc(),
            'post_conv1': {
                'weight': {
                    'shape': list(state_dict['post_conv1.weight'].shape),
                    'values': state_dict['post_conv1.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['post_conv1.bias'].shape),
                    'values': state_dict['post_conv1.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'post_conv2': {
                'weight': {
                    'shape': list(state_dict['post_conv2.weight'].shape),
                    'values': state_dict['post_conv2.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['post_conv2.bias'].shape),
                    'values': state_dict['post_conv2.bias'].flatten().cpu().numpy().tolist()
                }
            }
        }
        for i, block in enumerate(self.block_stack):
            doc['state_dict'][f'block_stack.{i}'] = block.generate_doc()
        return doc