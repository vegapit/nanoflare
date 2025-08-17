import torch
import torch.nn as nn
from .modules import BaseModel, PlainSequential

class HammersteinWeiner( BaseModel ):
    def __init__(self, input_size, hidden_size, output_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.f = nn.Tanh()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.linear_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        x: (batch_size, input_size, seq_len)
        """
        y = self.normalise( x ).transpose(1,2)
        y = self.input_linear( y )
        y = self.f( y )
        y, _ = self.linear_layer( y, hc )
        y = self.hidden_linear( y )
        y = self.f( y )
        return self.output_linear( y ).transpose(1,2)

    def generate_doc(self, meta_data={}):
        doc = {
            'config': {
                'model_type': 'HammersteinWeiner',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'meta_data': meta_data,
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_size': self.hidden_size
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
            'linear_layer': {
                'weight_hh_l0': {
                    'shape': list(state_dict['linear_layer.weight_hh_l0'].shape),
                    'values': state_dict['linear_layer.weight_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'weight_ih_l0': {
                    'shape': list(state_dict['linear_layer.weight_ih_l0'].shape),
                    'values': state_dict['linear_layer.weight_ih_l0'].flatten().cpu().numpy().tolist()
                },
                'bias_hh_l0' : {
                    'shape': list(state_dict['linear_layer.bias_hh_l0'].shape),
                    'values': state_dict['linear_layer.bias_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'bias_ih_l0' : {
                    'shape': list(state_dict['linear_layer.bias_ih_l0'].shape),
                    'values': state_dict['linear_layer.bias_ih_l0'].flatten().cpu().numpy().tolist()
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
            }
        }
        return doc