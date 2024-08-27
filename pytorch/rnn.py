import torch
import torch.nn as nn
from audiomodel import AudioModel, WaveShaper

class ResLSTM(AudioModel):
    def __init__(self, input_size, hidden_size, output_size, rnn_bias, linear_bias, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_bias = rnn_bias
        self.linear_bias = linear_bias
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bias=rnn_bias)
        self.linear = nn.Linear(hidden_size, output_size, bias=linear_bias)
    def forward(self, x: torch.Tensor, hc: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        y, _ = self.rnn( self.normalise(x).transpose(1,2), hc )
        return x + self.linear( y ).transpose(1,2)
    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'ResLSTM',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_size': self.hidden_size,
                'rnn_bias': self.rnn_bias,
                'linear_bias': self.linear_bias
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'rnn': {
                'weight_hh_l0': {
                    'shape': list(state_dict['rnn.weight_hh_l0'].shape),
                    'values': state_dict['rnn.weight_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'weight_ih_l0': {
                    'shape': list(state_dict['rnn.weight_ih_l0'].shape),
                    'values': state_dict['rnn.weight_ih_l0'].flatten().cpu().numpy().tolist()
                }
            },
            'linear': {
                'weight': {
                    'shape': list(state_dict['linear.weight'].shape),
                    'values': state_dict['linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        if self.rnn_bias:
            doc['state_dict']['rnn']['bias_hh_l0'] = {
                'shape': list(state_dict['rnn.bias_hh_l0'].shape),
                'values': state_dict['rnn.bias_hh_l0'].flatten().cpu().numpy().tolist()
            }
            doc['state_dict']['rnn']['bias_ih_l0'] = {
                'shape': list(state_dict['rnn.bias_ih_l0'].shape),
                'values': state_dict['rnn.bias_ih_l0'].flatten().cpu().numpy().tolist()
            }
        if self.linear_bias:
            doc['state_dict']['linear']['bias'] = {
                'shape': list(state_dict['linear.bias'].shape),
                'values': state_dict['linear.bias'].flatten().cpu().numpy().tolist()
            }
        return doc

class ResGRU(AudioModel):
    def __init__(self, input_size, hidden_size, output_size, rnn_bias, linear_bias, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_bias = rnn_bias
        self.linear_bias = linear_bias
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bias=rnn_bias)
        self.linear = nn.Linear(hidden_size, output_size, bias=linear_bias)
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn( self.normalise(x).transpose(1,2), h )
        return x + self.linear( y ).transpose(1,2)
    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'ResGRU',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_size': self.hidden_size,
                'rnn_bias': self.rnn_bias,
                'linear_bias': self.linear_bias
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'rnn': {
                'weight_hh_l0': {
                    'shape': list(state_dict['rnn.weight_hh_l0'].shape),
                    'values': state_dict['rnn.weight_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'weight_ih_l0': {
                    'shape': list(state_dict['rnn.weight_ih_l0'].shape),
                    'values': state_dict['rnn.weight_ih_l0'].flatten().cpu().numpy().tolist()
                }
            },
            'linear': {
                'weight': {
                    'shape': list(state_dict['linear.weight'].shape),
                    'values': state_dict['linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        if self.rnn_bias:
            doc['state_dict']['rnn']['bias_hh_l0'] = {
                'shape': list(state_dict['rnn.bias_hh_l0'].shape),
                'values': state_dict['rnn.bias_hh_l0'].flatten().cpu().numpy().tolist()
            }
            doc['state_dict']['rnn']['bias_ih_l0'] = {
                'shape': list(state_dict['rnn.bias_ih_l0'].shape),
                'values': state_dict['rnn.bias_ih_l0'].flatten().cpu().numpy().tolist()
            }
        if self.linear_bias:
            doc['state_dict']['linear']['bias'] = {
                'shape': list(state_dict['linear.bias'].shape),
                'values': state_dict['linear.bias'].flatten().cpu().numpy().tolist()
            }
        return doc

class ExtendedResGRU(AudioModel):
    def __init__(self, input_size, hidden_size, output_size, rnn_bias, linear_bias, ws_hidden_size, ws_num_layers, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn_bias = rnn_bias
        self.linear_bias = linear_bias
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, bias=rnn_bias)
        self.linear = nn.Linear(hidden_size, output_size, bias=linear_bias)
        self.ws = WaveShaper( ws_hidden_size, ws_num_layers )
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        norm_x = self.normalise(x).transpose(1,2)
        y = self.ws( norm_x )
        y, _ = self.rnn( y, h )
        return x + self.linear( y ).transpose(1,2)
    def generate_doc(self):
        doc = {
            'config': {
                'model_type': 'ResGRU',
                'norm_mean': self.norm_mean.item(),
                'norm_std': self.norm_std.item()
            },
            'parameters': {
                'input_size': self.input_size,
                'output_size': self.output_size,
                'hidden_size': self.hidden_size,
                'rnn_bias': self.rnn_bias,
                'linear_bias': self.linear_bias
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'rnn': {
                'weight_hh_l0': {
                    'shape': list(state_dict['rnn.weight_hh_l0'].shape),
                    'values': state_dict['rnn.weight_hh_l0'].flatten().cpu().numpy().tolist()
                },
                'weight_ih_l0': {
                    'shape': list(state_dict['rnn.weight_ih_l0'].shape),
                    'values': state_dict['rnn.weight_ih_l0'].flatten().cpu().numpy().tolist()
                }
            },
            'linear': {
                'weight': {
                    'shape': list(state_dict['linear.weight'].shape),
                    'values': state_dict['linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        if self.rnn_bias:
            doc['state_dict']['rnn']['bias_hh_l0'] = {
                'shape': list(state_dict['rnn.bias_hh_l0'].shape),
                'values': state_dict['rnn.bias_hh_l0'].flatten().cpu().numpy().tolist()
            }
            doc['state_dict']['rnn']['bias_ih_l0'] = {
                'shape': list(state_dict['rnn.bias_ih_l0'].shape),
                'values': state_dict['rnn.bias_ih_l0'].flatten().cpu().numpy().tolist()
            }
        if self.linear_bias:
            doc['state_dict']['linear']['bias'] = {
                'shape': list(state_dict['linear.bias'].shape),
                'values': state_dict['linear.bias'].flatten().cpu().numpy().tolist()
            }
        return doc
    
if __name__ == "__main__":
    model = ResLSTM(1, 5, 1, False, False)

    model.eval()

    x = torch.randn((5, 1, 1024)).to(torch.float32)
    h = torch.zeros((1, 5, 5)).to(torch.float32)
    c = torch.zeros((1, 5, 5)).to(torch.float32)
    y = model( x, (h, c) )
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )