import torch
import torch.nn as nn
from .modules import BaseModel, TCNBlock, PlainSequential

class HammersteinWiener( BaseModel ):
    def __init__(self, input_size, linear_input_size, linear_output_size, kernel_size, stack_size, hidden_size, output_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.linear_input_size = linear_input_size
        self.linear_output_size = linear_output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Nonlinear input stage (Hammerstein)
        self.input_linear = nn.Linear(input_size, linear_input_size)
        self.f_in = nn.LeakyReLU(negative_slope=0.2)
        
        # Dynamic linear stage (memory)
        self.block_stack = nn.ModuleList([
            TCNBlock(
                linear_input_size if i == 0 else linear_output_size,
                linear_output_size,
                kernel_size,
                2**i) # Dilation
            for i in range(stack_size)
        ])

        # Nonlinear output stage (Wiener)
        self.hidden_linear = nn.Linear(linear_output_size, hidden_size)
        self.f_out = nn.LeakyReLU(negative_slope=0.2)
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Identity skip path
        self.skip_linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_size, seq_len)
        """
        x_t = x.transpose(1,2)  # (batch, time, features)
        y = self.normalise( x_t )
        y = self.f_in( self.input_linear( y ) )

        y = y.transpose(1, 2)  # (batch, features, time)
        for block in self.block_stack:
            y = block( y )
        y = y.transpose(1, 2)

        y = self.f_out( self.hidden_linear( y ) )
        # Residual skip: dry passthrough + learned coloration
        return self.skip_linear( x_t ) + self.output_linear( y )

    def generate_doc(self, meta_data={}):
        doc = {
            'config': {
                'model_type': 'HammersteinWiener',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'meta_data': meta_data,
            'parameters': {
                'input_size': self.input_size,
                'linear_input_size': self.linear_input_size,
                'linear_output_size': self.linear_output_size,
                'kernel_size': self.kernel_size,
                'stack_size': self.stack_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'input_linear': {
                'weight': {
                    'shape': list(state_dict['input_linear.weight'].shape),
                    'values': state_dict['input_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['input_linear.bias'].shape),
                    'values': state_dict['input_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'hidden_linear': {
                'weight': {
                    'shape': list(state_dict['hidden_linear.weight'].shape),
                    'values': state_dict['hidden_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['hidden_linear.bias'].shape),
                    'values': state_dict['hidden_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'output_linear': {
                'weight': {
                    'shape': list(state_dict['output_linear.weight'].shape),
                    'values': state_dict['output_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['output_linear.bias'].shape),
                    'values': state_dict['output_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'skip_linear': {
                'weight': {
                    'shape': list(state_dict['skip_linear.weight'].shape),
                    'values': state_dict['skip_linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        for i, block in enumerate(self.block_stack):
            doc['state_dict'][f'block_stack.{i}'] = block.generate_doc()
        return doc