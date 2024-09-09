import torch
import torch.nn as nn
from audiomodel import AudioModel, CausalDilatedConv1d, ResidualBlock, PlainSequential

class WaveNet(AudioModel):
    def __init__(self, input_size, num_channels, output_size, kernel_size, dilations, stack_size, gated, ps_hidden_size, ps_num_hidden_layers, norm_mean=0.0, norm_std=1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.num_channels = num_channels
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.stack_size = stack_size
        self.gated = gated
        self.conv = CausalDilatedConv1d(input_size, num_channels, kernel_size)
        self.block_stack = nn.ModuleList([ ResidualBlock(num_channels, kernel_size, d, gated) for d in dilations ] * stack_size)
        self.plain_sequential = PlainSequential( num_channels, output_size, ps_hidden_size, ps_num_hidden_layers )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.normalise(x)
        y = self.conv( x )
        skip_sum = torch.zeros_like( y )
        for block in self.block_stack:
            y, skip_y = block(y)
            skip_sum += skip_y
        return self.plain_sequential( self.relu(skip_sum).transpose(1,2) ).transpose(1,2)

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
                'ps_hidden_size': self.plain_sequential.hidden_size,
                'ps_num_hidden_layers': self.plain_sequential.num_hidden_layers
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'conv': self.conv.generate_doc(),
            'plain_sequential': self.plain_sequential.generate_doc()
        }
        for i, block in enumerate(self.block_stack):
            doc['state_dict'][f'block_stack.{i}'] = block.generate_doc()
        return doc
        
if __name__ == "__main__":
    model = WaveNet(1, 8, 1, 3, [1, 2, 4, 8, 16], 2, False)

    model.eval()

    x = torch.randn((1, 1, 1024)).to(torch.float32)
    y = model( x )
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )