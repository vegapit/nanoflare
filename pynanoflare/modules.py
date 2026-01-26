import torch
from torch import nn

class BaseModel( nn.Module ):
    def __init__(self, norm_mean: float, norm_std: float):
        super().__init__()
        self.register_buffer('norm_mean', torch.tensor([norm_mean]))
        self.register_buffer('norm_std', torch.tensor([norm_std]))
    def normalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.norm_mean) / self.norm_std
    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.norm_std + self.norm_mean

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation # Add required padding on both sides of the input
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(x, (self.padding, 0))
        return self.conv1d(x)
        
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

class FiLM( nn.Module ):
    def __init__(self, feature_dim, control_dim):
        super().__init__()
        self.scale = nn.Linear(control_dim, feature_dim)
        self.shift = nn.Linear(control_dim, feature_dim)
    def forward(self, x, params):
        # x : [batch,time,features]
        gamma = self.scale(params).unsqueeze(1)
        beta  = self.shift(params).unsqueeze(1)
        return x * gamma + beta
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
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_batchnorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1 = CausalDilatedConv1d( in_channels, out_channels, kernel_size, dilation=dilation )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.f1 = nn.LeakyReLU( 0.2, inplace=True )

    def forward(self, x: torch.Tensor):
        y = self.conv1( x )
        if(self.use_batchnorm):
            y = self.bn1( y )
        self.f1( y )
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
        self.direct_linear = nn.Linear( input_size, output_size, False )
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
        # Dilated causal conv expands features
        conv_out_channels = num_channels * 2 if gated else num_channels
        self.input_conv = CausalDilatedConv1d(num_channels, conv_out_channels, kernel_size, dilation)
        # 1x1 convs for residual and skip projections
        self.residual_conv = nn.Conv1d(num_channels, num_channels, 1)
        self.skip_conv = nn.Conv1d(num_channels, num_channels, 1)
        # Nonlinearities
        self.f = nn.Tanh()
        self.g = nn.Sigmoid() 
        
    def forward(self, x : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Dilated causal conv
        conv_out = self.input_conv(x)
        # Gated activation or plain Tanh
        if self.gated:
            y_f, y_g = conv_out.chunk(2, dim=1) # Separate Filter and Gate
            z = self.f( y_f ) * self.g( y_g )
        else:
            z = self.f( conv_out )
        return x + self.residual_conv( z ), self.skip_conv( z )
    
    def generate_doc(self):
        state_dict = self.state_dict()
        doc = {
            'input_conv': self.input_conv.generate_doc(),
            'residual_conv': {
                'weight': {
                    'shape': list(state_dict['residual_conv.weight'].shape),
                    'values': state_dict['residual_conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['residual_conv.bias'].shape),
                    'values': state_dict['residual_conv.bias'].flatten().cpu().numpy().tolist()
                }
            },
            'skip_conv': {
                'weight': {
                    'shape': list(state_dict['skip_conv.weight'].shape),
                    'values': state_dict['skip_conv.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict['skip_conv.bias'].shape),
                    'values': state_dict['skip_conv.bias'].flatten().cpu().numpy().tolist()
                }
            }
        }
        return doc

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, use_batchnorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_batchnorm = use_batchnorm
        # Residual connection if in_channels != out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        # Two layers of dilated causal convolution
        self.conv1 = CausalDilatedConv1d( in_channels, out_channels, kernel_size, dilation=dilation )
        self.conv2 = CausalDilatedConv1d( out_channels, out_channels, kernel_size, dilation=1 )
        # Normalization (optional)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Activations
        self.f1 = nn.LeakyReLU( 0.2 )
        self.f2 = nn.LeakyReLU( 0.2 )

    def forward(self, x: torch.Tensor):
        
        y = self.conv1( x )
        if(self.use_batchnorm):
            y = self.bn1( y )
        y = self.f1( y )

        y = self.conv2( y )
        if(self.use_batchnorm):
            y = self.bn2( y )
        y = self.f2( y )

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
            }
        }
        return doc

class Biquad(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.randn(5) * 0.1)
        # Register buffers for state (z1, z2 per channel)
        # State will be initialized in forward based on input channels

    def extract_coefs(self):
        """Converts stable polar coordinates to biquad coefficients."""
        # Force pole radius to be < 1.0 for absolute stability
        p_r = torch.tanh(self.params[0]) * 0.98
        p_theta = torch.sigmoid(self.params[1]) * torch.pi
        z_r = torch.tanh(self.params[2]) * 0.99
        z_theta = torch.sigmoid(self.params[3]) * torch.pi
        gain = torch.exp(self.params[4])

        # Standard conversion from Polar to biquad coefficients
        a1 = -2 * p_r * torch.cos(p_theta)
        a2 = p_r ** 2
        b0 = gain
        b1 = -2 * gain * z_r * torch.cos(z_theta)
        b2 = gain * z_r ** 2

        return b0, b1, b2, 1.0, a1, a2

    def forward(self, x):
        """
        Apply biquad filter using Direct Form II Transposed.
        Supports both 2D [channels, samples] and 3D [batch, channels, samples] inputs.
        """
        b0, b1, b2, a0, a1, a2 = self.extract_coefs()

        # Manual biquad implementation
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        # Direct Form II Transposed: y[n] = b0*x[n] + z1[n-1]
        #                             z1[n] = b1*x[n] + z2[n-1] - a1*y[n]
        #                             z2[n] = b2*x[n] - a2*y[n]

        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # 2D input: [channels, samples]
            channels, samples = x.shape
            batch_size = 1
            # Add batch dimension for uniform processing
            x = x.unsqueeze(0)  # [1, channels, samples]
            was_2d = True
        elif x.dim() == 3:
            # 3D input: [batch, channels, samples]
            batch_size, channels, samples = x.shape
            was_2d = False
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.dim()}D")

        y = torch.zeros_like(x)

        # State variables (delay line) with shape [batch, channels]
        z1 = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)
        z2 = torch.zeros(batch_size, channels, device=x.device, dtype=x.dtype)

        for n in range(samples):
            # Process all batches and channels simultaneously
            # Use .clone() to avoid in-place operation issues
            y_n = b0 * x[:, :, n] + z1
            y[:, :, n] = y_n
            
            # Update state variables
            z1 = b1 * x[:, :, n] + z2 - a1 * y_n
            z2 = b2 * x[:, :, n] - a2 * y_n

        # Remove batch dimension if input was 2D
        if was_2d:
            y = y.squeeze(0)  # [channels, samples]

        return y

    def generate_doc(self):
        # Compute final biquad coefficients for export
        b0, b1, b2, a0, a1, a2 = self.extract_coefs()

        # Convert to numpy and export as list
        doc = {
            'b0': float(b0.item()),
            'b1': float(b1.item()),
            'b2': float(b2.item()),
            'a0': float(a0),  # Always 1.0
            'a1': float(a1.item()),
            'a2': float(a2.item())
        }
        return doc
