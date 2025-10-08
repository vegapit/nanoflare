import torch
import torch.nn as nn
from .modules import BaseModel, PlainSequential

class HammersteinWiener( BaseModel ):
    def __init__(self, input_size, linear_input_size, linear_output_size, hidden_size, output_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.linear_input_size = linear_input_size
        self.linear_output_size = linear_output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Nonlinear input stage (Hammerstein)
        self.input_linear = nn.Linear(input_size, linear_input_size)
        self.f_in = nn.Tanh()
        
        # Dynamic linear stage (memory)
        self.lstm = nn.LSTM(
            input_size=linear_input_size,
            hidden_size=linear_output_size,
            num_layers=1,
            batch_first=True
        )

        # Nonlinear output stage (Wiener)
        self.hidden_linear = nn.Linear(linear_output_size, hidden_size)
        self.f_out = nn.Tanh()
        self.output_linear = nn.Linear(hidden_size, output_size)

        # Identity skip path
        self.skip_linear = nn.Linear(input_size, output_size, bias=False)
         # initialize as identity and scale down
        nn.init.eye_(self.skip_linear.weight) 
        self.skip_linear.weight *= 0.5

    def forward(self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        x: (batch_size, input_size, seq_len)
        """
        y = self.normalise( x ).transpose(1,2)
        y = self.f_in( self.input_linear( y ) )
        y, hc = self.lstm( y, hc )
        y = self.f_out( self.hidden_linear( y ) )
        y = self.output_linear( y ).transpose(1,2)
        # Residual skip: dry passthrough + learned coloration
        return self.skip_linear( x.transpose(1,2) ).transpose(1,2) + y, hc

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
            'lstm': {
                'weight_hh_l0': {
                    'shape': list(state_dict['lstm.weight_hh_l0'].shape),
                    'values': state_dict['lstm.weight_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'weight_ih_l0': {
                    'shape': list(state_dict['lstm.weight_ih_l0'].shape),
                    'values': state_dict['lstm.weight_ih_l0'].flatten().cpu().numpy().tolist()
                },
                'bias_hh_l0' : {
                    'shape': list(state_dict['lstm.bias_hh_l0'].shape),
                    'values': state_dict['lstm.bias_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'bias_ih_l0' : {
                    'shape': list(state_dict['lstm.bias_ih_l0'].shape),
                    'values': state_dict['lstm.bias_ih_l0'].flatten().cpu().numpy().tolist()
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
        return doc