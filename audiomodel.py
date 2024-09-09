import torch
import torch.nn as nn

class AudioModel(nn.Module):
    def __init__(self, norm_mean: float, norm_std: float):
        super().__init__()
        self.norm_mean = torch.nn.parameter.Parameter( data= torch.FloatTensor([norm_mean]), requires_grad=False )
        self.norm_std = torch.nn.parameter.Parameter( data= torch.FloatTensor([norm_std]), requires_grad=False )
    @torch.no_grad()
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std

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
                    'shape': list(state_dict[f'direct_linear.weight'].shape),
                    'values': state_dict[f'direct_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'direct_linear.bias'].shape),
                    'values': state_dict[f'direct_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'input_linear': {
                'weight': {
                    'shape': list(state_dict[f'input_linear.weight'].shape),
                    'values': state_dict[f'input_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'input_linear.bias'].shape),
                    'values': state_dict[f'input_linear.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'output_linear': {
                'weight': {
                    'shape': list(state_dict[f'output_linear.weight'].shape),
                    'values': state_dict[f'output_linear.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'output_linear.bias'].shape),
                    'values': state_dict[f'output_linear.bias'].flatten().cpu().numpy().tolist()
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
                'shape': list(state_dict[f'conv1d.weight'].shape),
                'values': state_dict[f'conv1d.weight'].flatten().cpu().numpy().tolist()
            },
            'bias': {
                'shape': list(state_dict[f'conv1d.bias'].shape),
                'values': state_dict[f'conv1d.bias'].flatten().cpu().numpy().tolist()
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
                    'shape': list(state_dict[f'output_conv.weight'].shape),
                    'values': state_dict[f'output_conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'output_conv.bias'].shape),
                    'values': state_dict[f'output_conv.bias'].flatten().cpu().numpy().tolist()
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
                    'shape': list(state_dict[f'conv.weight'].shape),
                    'values': state_dict[f'conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'conv.bias'].shape),
                    'values': state_dict[f'conv.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'conv1': self.conv1.generate_doc(),
            'conv2': self.conv2.generate_doc(),
            'bn1': {
                'weight': {
                    'shape': list(state_dict[f'bn1.weight'].shape),
                    'values': state_dict[f'bn1.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'bn1.bias'].shape),
                    'values': state_dict[f'bn1.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict[f'bn1.running_mean'].shape),
                    'values': state_dict[f'bn1.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict[f'bn1.running_var'].shape),
                    'values': state_dict[f'bn1.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'bn2': {
                'weight': {
                    'shape': list(state_dict[f'bn2.weight'].shape),
                    'values': state_dict[f'bn2.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'bn2.bias'].shape),
                    'values': state_dict[f'bn2.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict[f'bn2.running_mean'].shape),
                    'values': state_dict[f'bn2.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict[f'bn2.running_var'].shape),
                    'values': state_dict[f'bn2.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'f1': {
                'weight' : {
                    'shape': list(state_dict[f'f1.weight'].shape),
                    'values': state_dict[f'f1.weight'].flatten().cpu().numpy().tolist()
                }
            },
            'f2': {
                'weight' : {
                    'shape': list(state_dict[f'f2.weight'].shape),
                    'values': state_dict[f'f2.weight'].flatten().cpu().numpy().tolist()
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
                    'shape': list(state_dict[f'conv.weight'].shape),
                    'values': state_dict[f'conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'conv.bias'].shape),
                    'values': state_dict[f'conv.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'conv1': self.conv1.generate_doc(),
            'bn1': {
                'weight': {
                    'shape': list(state_dict[f'bn1.weight'].shape),
                    'values': state_dict[f'bn1.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'bn1.bias'].shape),
                    'values': state_dict[f'bn1.bias'].flatten().cpu().numpy().tolist()
                },
                'running_mean': {
                    'shape': list(state_dict[f'bn1.running_mean'].shape),
                    'values': state_dict[f'bn1.running_mean'].flatten().cpu().numpy().tolist()
                },
                'running_var': {
                    'shape': list(state_dict[f'bn1.running_var'].shape),
                    'values': state_dict[f'bn1.running_var'].flatten().cpu().numpy().tolist()
                }
            },
            'f1': {
                'weight' : {
                    'shape': list(state_dict[f'f1.weight'].shape),
                    'values': state_dict[f'f1.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc
    
def sr_loss(y, y_pred, coeff=0.85):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y, coeff=coeff), pre_emphasis_filter(y_pred, coeff=coeff)
    return (y - y_pred).pow(2).sum() / y.pow(2).sum()

def dc_loss(y, y_pred):
    """
    DC offset loss
    """
    return (y - y_pred).mean()**2.0 / y.pow(2).mean()

def snr_loss(y, y_pred):
    return (y - y_pred).var()

def pre_emphasis_filter(x, coeff=0.85): # for n >= 1, y[n] = x[n] - 0.85 * x[n-1] or y[0] = x[0]
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)