import torch
import torch.nn as nn
from .modules import BaseModel, TCNBlock, MicroTCNBlock, PlainSequential

class TCN( BaseModel ):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.block_stack = nn.ModuleList([TCNBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 2**i) for i in range(stack_size)])
        self.plain_sequential = PlainSequential( hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalise( x )
        for block in self.block_stack:
            x = block( x )
        return self.plain_sequential( x.transpose(1,2) ).transpose(1,2)

    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'TCN',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'kernel_size': self.kernel_size,
                'stack_size': self.stack_size,
                'ps_hidden_size': self.plain_sequential.hidden_size,
                'ps_num_hidden_layers': self.plain_sequential.num_hidden_layers
            }
        }
        doc['state_dict'] = {
            'plain_sequential': self.plain_sequential.generate_doc()
        }
        for i, block in enumerate(self.block_stack):
            doc['state_dict'][f'block_stack.{i}'] = block.generate_doc()
        return doc

class MicroTCN( BaseModel ):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.block_stack = nn.ModuleList([MicroTCNBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 2**i) for i in range(stack_size)])
        self.plain_sequential = PlainSequential( hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalise( x )
        for block in self.block_stack:
            x = block( x )
        return self.plain_sequential( x.transpose(1,2) ).transpose(1,2)
    
    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'MicroTCN',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'kernel_size': self.kernel_size,
                'stack_size': self.stack_size,
                'ps_hidden_size': self.plain_sequential.hidden_size,
                'ps_num_hidden_layers': self.plain_sequential.num_hidden_layers
            }
        }
        doc['state_dict'] = {
            'plain_sequential' : self.plain_sequential.generate_doc()
        }
        for i, block in enumerate(self.block_stack):
            doc['state_dict'][f'block_stack.{i}'] = block.generate_doc()
        return doc
    
if __name__ == "__main__":
    model = TCN(1, 1, 5, 5)

    model.eval()

    x = torch.randn((1, 1, 1024)).to(torch.float32)
    y = model( x )
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )