import torch
from torch import nn

class BaseModel( nn.Module ):
    def __init__(self, norm_mean: float, norm_std: float):
        super().__init__()
        self.norm_mean = torch.nn.parameter.Parameter( data= torch.FloatTensor([norm_mean]), requires_grad=False )
        self.norm_std = torch.nn.parameter.Parameter( data= torch.FloatTensor([norm_std]), requires_grad=False )
    @torch.no_grad()
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std
    
class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation # Add required padding on both sides of the input
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        y = self.conv1d(x)
        return y[ ..., :-self.padding] # Add remove padding on right-hande side of the output
        
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'weight': {
                'shape': list(state_dict['conv1d.weight'].shape),
                'values': state_dict['conv1d.weight'].flatten().cpu().numpy().tolist()
            },
            'bias': {
                'shape': list(state_dict['conv1d.bias'].shape),
                'values': state_dict['conv1d.bias'].flatten().cpu().numpy().tolist()
            }
        }
        return doc

class ConvClipper( nn.Module ):
    def __init__(self, input_size, output_size, kernel_size, dilation):
        super().__init__()
        self.conv = CausalDilatedConv1d(input_size, output_size, kernel_size, dilation)
        self.floor = nn.parameter.Parameter( torch.zeros(1), requires_grad=True )
        self.ceiling = nn.parameter.Parameter( torch.zeros(1), requires_grad=True )
        self.coef_softsign = nn.parameter.Parameter( torch.randn(1), requires_grad=True )
        self.coef_tanh = nn.parameter.Parameter( torch.randn(1), requires_grad=True )
    def forward(self, x):
        y = self.conv( x )
        y = y + nn.functional.softsign( self.coef_softsign  * y )
        y = y + nn.functional.tanh( self.coef_tanh * y )
        return torch.clip( y, min=-torch.sigmoid( 5.0 * self.floor ), max=torch.sigmoid( 5.0 * self.ceiling ) )
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'conv': self.conv.generate_doc(),
            'floor': {
                'shape': list(state_dict['floor'].shape),
                'values': state_dict['floor'].flatten().cpu().numpy().tolist()
            },
            'ceiling': {
                'shape': list(state_dict['ceiling'].shape),
                'values': state_dict['ceiling'].flatten().cpu().numpy().tolist()
            },
            'coef_softsign': {
                'shape': list(state_dict['coef_softsign'].shape),
                'values': state_dict['coef_softsign'].flatten().cpu().numpy().tolist()
            },
            'coef_tanh': {
                'shape': list(state_dict['coef_tanh'].shape),
                'values': state_dict['coef_tanh'].flatten().cpu().numpy().tolist()
            }
        }
        return doc

class FiLM( nn.Module ):
    def __init__(self, feature_dim, control_dim):
        super().__init__()
        self.scale = nn.Linear(control_dim, feature_dim)
        self.shift = nn.Linear(control_dim, feature_dim)
    def forward(self, x, params):
        return x * self.scale( params ) + self.shift( params )
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'scale': {
                'weight': {
                    'shape': list(state_dict['scale.weight'].shape),
                    'values': state_dict['scale.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['scale.bias'].shape),
                    'values': state_dict['scale.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'shift': {
                'weight': {
                    'shape': list(state_dict['shift.weight'].shape),
                    'values': state_dict['shift.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['shift.bias'].shape),
                    'values': state_dict['shift.bias'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc

class MicroTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1 = CausalDilatedConv1d( in_channels, out_channels, kernel_size, dilation=dilation )
        self.f1 = nn.PReLU( out_channels )
        self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        y = self.bn1(self.f1(self.conv1(x)))
        if(self.in_channels == self.out_channels):
            return x + y
        else:
            return self.conv(x) + y
    
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'conv': {
                'weight': {
                    'shape': list(state_dict['conv.weight'].shape),
                    'values': state_dict['conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['conv.bias'].shape),
                    'values': state_dict['conv.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'conv1': self.conv1.generate_doc(),
            'bn1': {
                'weight': {
                    'shape': list(state_dict['bn1.weight'].shape),
                    'values': state_dict['bn1.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['bn1.bias'].shape),
                    'values': state_dict['bn1.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict['bn1.running_mean'].shape),
                    'values': state_dict['bn1.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict['bn1.running_var'].shape),
                    'values': state_dict['bn1.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'f1': {
                'weight' : {
                    'shape': list(state_dict['f1.weight'].shape),
                    'values': state_dict['f1.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc
        
class PlainSequential( nn.Module ):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.direct_linear = nn.Linear( input_size, output_size )
        self.input_linear = nn.Linear( input_size, hidden_size )
        self.hidden_linear = nn.ModuleList( [ nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers) ] )
        self.output_linear = nn.Linear(hidden_size, output_size)
        self.f = nn.ReLU()
    def forward(self, x):
        y = self.f( self.input_linear( x ) )
        for layer in self.hidden_linear:
            y = self.f( layer( y ) )
        if self.input_size == self.output_size:
            return x + self.output_linear( y )
        else:
            return self.direct_linear( x ) + self.output_linear( y )
    
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'direct_linear': {
                'weight': {
                    'shape': list(state_dict['direct_linear.weight'].shape),
                    'values': state_dict['direct_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['direct_linear.bias'].shape),
                    'values': state_dict['direct_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
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
        for i, _ in enumerate(self.hidden_linear):
            doc[f'hidden_linear.{i}'] = {
                'weight': {
                    'shape': list(state_dict[f'hidden_linear.{i}.weight'].shape),
                    'values': state_dict[f'hidden_linear.{i}.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'hidden_linear.{i}.bias'].shape),
                    'values': state_dict[f'hidden_linear.{i}.bias'].flatten().cpu().numpy().tolist()
                }
            }
        return doc

class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, dilation, gated):
        super().__init__()
        self.num_channels = num_channels
        self.gated = gated
        self.input_conv = CausalDilatedConv1d(num_channels, 2 * num_channels if gated else num_channels, kernel_size, dilation=dilation)
        self.output_conv = nn.Conv1d(num_channels, num_channels, 1)
        self.f = nn.Tanh()
        self.g = nn.Sigmoid() # Gate activation function
        
    def forward(self, x):
        if self.gated:
            ys = torch.split( self.input_conv(x), self.num_channels, dim=1) # Separate Filter and Gate
            y = self.f( ys[0] ) * self.g( ys[1] )
        else:
            y = self.f( self.input_conv(x) )
        y = self.output_conv( y )
        return y + x, y
    
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'input_conv': self.input_conv.generate_doc(),
            'output_conv': {
                'weight': {
                    'shape': list(state_dict['output_conv.weight'].shape),
                    'values': state_dict['output_conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['output_conv.bias'].shape),
                    'values': state_dict['output_conv.bias'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1 = CausalDilatedConv1d( in_channels, out_channels, kernel_size, dilation=dilation )
        self.f1 = nn.PReLU( out_channels )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalDilatedConv1d( out_channels, out_channels, kernel_size, dilation=1 )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.f2 = nn.PReLU( out_channels )

    def forward(self, x: torch.Tensor):
        y = self.bn1(self.f1(self.conv1(x)))
        y = self.bn2(self.f2(self.conv2(y)))
        if(self.in_channels == self.out_channels):
            return x + y
        else:
            return self.conv(x) + y
        
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'conv': {
                'weight': {
                    'shape': list(state_dict['conv.weight'].shape),
                    'values': state_dict['conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['conv.bias'].shape),
                    'values': state_dict['conv.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'conv1': self.conv1.generate_doc(),
            'conv2': self.conv2.generate_doc(),
            'bn1': {
                'weight': {
                    'shape': list(state_dict['bn1.weight'].shape),
                    'values': state_dict['bn1.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['bn1.bias'].shape),
                    'values': state_dict['bn1.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict['bn1.running_mean'].shape),
                    'values': state_dict['bn1.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict['bn1.running_var'].shape),
                    'values': state_dict['bn1.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'bn2': {
                'weight': {
                    'shape': list(state_dict['bn2.weight'].shape),
                    'values': state_dict['bn2.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['bn2.bias'].shape),
                    'values': state_dict['bn2.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict['bn2.running_mean'].shape),
                    'values': state_dict['bn2.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict['bn2.running_var'].shape),
                    'values': state_dict['bn2.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'f1': {
                'weight' : {
                    'shape': list(state_dict['f1.weight'].shape),
                    'values': state_dict['f1.weight'].flatten().cpu().numpy().tolist()
                }
            },
            'f2': {
                'weight' : {
                    'shape': list(state_dict['f2.weight'].shape),
                    'values': state_dict['f2.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc