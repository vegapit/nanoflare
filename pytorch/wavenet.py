import torch
import torch.nn as nn
from audiomodel import AudioModel, CausalDilatedConv1d, ResidualBlock

class WaveNet(AudioModel):
    def __init__(self, input_size, num_channels, output_size, kernel_size, dilations, stack_size, gated, activation, norm_mean=0.0, norm_std=1.0):
        assert( activation in ["Sigmoid","Tanh","SoftSign"] )
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.num_channels = num_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.stack_size = stack_size
        self.gated = gated
        self.activation = activation
        self.conv = CausalDilatedConv1d(1, num_channels, kernel_size)
        self.blockStack = nn.ModuleList([ ResidualBlock(num_channels, kernel_size, d, gated, activation) for d in dilations ] * stack_size)
        self.linear = nn.Linear(num_channels, 1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        #print(f"WaveNet: {x.shape}")
        x = self.normalise(x)
        y = self.conv(x)
        skip_sum = torch.zeros_like(y)
        for block in self.blockStack:
            y, skip_y = block(y)
            skip_sum += skip_y
        return self.linear( self.relu(skip_sum).transpose(1,2) ).transpose(1,2)

    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'WaveNet',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'num_channels': self.num_channels,
                'kernel_size': self.kernel_size,
                'dilations': self.dilations,
                'stack_size': self.stack_size,
                'gated': self.gated,
                'activation': self.activation
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'conv': {
                'weight': {
                    'shape': list(state_dict['conv.conv1d.weight'].shape),
                    'values': state_dict['conv.conv1d.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['conv.conv1d.bias'].shape),
                    'values': state_dict['conv.conv1d.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'linear': {
                'weight': {
                    'shape': list(state_dict['linear.weight'].shape),
                    'values': state_dict['linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        for i,block in enumerate(self.blockStack):
            doc['state_dict'][f'blockStack.{i}'] = {
                'inputConv': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.inputConv.conv1d.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.inputConv.conv1d.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.inputConv.conv1d.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.inputConv.conv1d.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'outputConv': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.outputConv.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.outputConv.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.outputConv.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.outputConv.bias'].flatten().cpu().numpy().tolist()
                    }
                }
            }
        return doc
        
if __name__ == "__main__":
    model = WaveNet(1, 8, 1, 3, [1, 2, 4, 8, 16], 2, False, "Tanh")

    model.eval()

    x = torch.randn((1, 1, 1024)).to(torch.float32)
    y = model( x )
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )