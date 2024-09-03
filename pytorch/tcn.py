import torch
import torch.nn as nn
from audiomodel import AudioModel, TCNBlock, MicroTCNBlock, PlainSequential

class TCN(AudioModel):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, stack_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.blockStack = nn.ModuleList([TCNBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 2**i) for i in range(stack_size)])
        self.linear = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalise( x )
        for block in self.blockStack:
            x = block( x )
        return self.linear( x.transpose(1,2) ).transpose(1,2)

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
                'stack_size': self.stack_size
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'linear': {
                'weight': {
                    'shape': list(state_dict['linear.weight'].shape),
                    'values': state_dict['linear.weight'].flatten().cpu().numpy().tolist()
                }
            }
        }
        for i,_ in enumerate(self.blockStack):
            doc['state_dict'][f'blockStack.{i}'] = {
                'conv': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.conv.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.conv.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.conv.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.conv.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'conv1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.conv1.conv1d.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.conv1.conv1d.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.conv1.conv1d.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.conv1.conv1d.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'bn1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.bias'].flatten().cpu().numpy().tolist()
                    },
                    'running_mean': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.running_mean'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.running_mean'].flatten().cpu().numpy().tolist()
                    },
                    'running_var': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.running_var'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.running_var'].flatten().cpu().numpy().tolist()
                    }
                },
                'conv2': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.conv2.conv1d.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.conv2.conv1d.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.conv2.conv1d.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.conv2.conv1d.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'bn2': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.bn2.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.bn2.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.bn2.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.bn2.bias'].flatten().cpu().numpy().tolist()
                    },
                    'running_mean': {
                        'shape': list(state_dict[f'blockStack.{i}.bn2.running_mean'].shape),
                        'values': state_dict[f'blockStack.{i}.bn2.running_mean'].flatten().cpu().numpy().tolist()
                    },
                    'running_var': {
                        'shape': list(state_dict[f'blockStack.{i}.bn2.running_var'].shape),
                        'values': state_dict[f'blockStack.{i}.bn2.running_var'].flatten().cpu().numpy().tolist()
                    }
                },
                'f1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.f1.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.f1.weight'].flatten().cpu().numpy().tolist()
                    }
                },
                'f2': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.f2.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.f2.weight'].flatten().cpu().numpy().tolist()
                    }
                }
            }
        return doc

class MicroTCN(AudioModel):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, stack_size, ps_hidden_size, ps_num_hidden_layers, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.ps_hidden_size = ps_hidden_size
        self.ps_num_hidden_layers = ps_num_hidden_layers
        self.blockStack = nn.ModuleList([MicroTCNBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, 2**i) for i in range(stack_size)])
        self.plain_sequential = PlainSequential( hidden_size, output_size, ps_hidden_size, ps_num_hidden_layers )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalise( x )
        for block in self.blockStack:
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
                'ps_hidden_size' : self.ps_hidden_size,
                'ps_num_hidden_layers': self.ps_num_hidden_layers 
            }
        }
        state_dict = self.state_dict()
        doc['state_dict'] = {
            'plain_sequential' : {
                'input_linear': {
                    'weight': {
                        'shape': list(state_dict[f'plain_sequential.input_linear.weight'].shape),
                        'values': state_dict[f'plain_sequential.input_linear.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'plain_sequential.input_linear.bias'].shape),
                        'values': state_dict[f'plain_sequential.input_linear.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'output_linear': {
                    'weight': {
                        'shape': list(state_dict[f'plain_sequential.output_linear.weight'].shape),
                        'values': state_dict[f'plain_sequential.output_linear.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'plain_sequential.output_linear.bias'].shape),
                        'values': state_dict[f'plain_sequential.output_linear.bias'].flatten().cpu().numpy().tolist()
                    }
                }
            }
        }
        for i, _ in enumerate(self.plain_sequential.hidden_linear):
            doc['state_dict']['plain_sequential'][f'hidden_linear.{i}'] = {
                'weight': {
                    'shape': list(state_dict[f'plain_sequential.hidden_linear.{i}.weight'].shape),
                    'values': state_dict[f'plain_sequential.hidden_linear.{i}.weight'].flatten().cpu().numpy().tolist()
                },
                'bias': {
                    'shape': list(state_dict[f'plain_sequential.hidden_linear.{i}.bias'].shape),
                    'values': state_dict[f'plain_sequential.hidden_linear.{i}.bias'].flatten().cpu().numpy().tolist()
                }
            }
        for i,_ in enumerate(self.blockStack):
            doc['state_dict'][f'blockStack.{i}'] = {
                'conv': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.conv.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.conv.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.conv.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.conv.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'conv1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.conv1.conv1d.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.conv1.conv1d.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.conv1.conv1d.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.conv1.conv1d.bias'].flatten().cpu().numpy().tolist()
                    }
                },
                'bn1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.weight'].flatten().cpu().numpy().tolist()
                    },
                    'bias': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.bias'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.bias'].flatten().cpu().numpy().tolist()
                    },
                    'running_mean': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.running_mean'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.running_mean'].flatten().cpu().numpy().tolist()
                    },
                    'running_var': {
                        'shape': list(state_dict[f'blockStack.{i}.bn1.running_var'].shape),
                        'values': state_dict[f'blockStack.{i}.bn1.running_var'].flatten().cpu().numpy().tolist()
                    }
                },
                'f1': {
                    'weight': {
                        'shape': list(state_dict[f'blockStack.{i}.f1.weight'].shape),
                        'values': state_dict[f'blockStack.{i}.f1.weight'].flatten().cpu().numpy().tolist()
                    }
                },
            }
        return doc
    
if __name__ == "__main__":
    model = TCN(1, 1, 5, 5)

    model.eval()

    x = torch.randn((1, 1, 1024)).to(torch.float32)
    y = model( x )
    print( f"input: {x.shape}")
    print( f"output: {y.shape}" )

    print( model.generate_doc() )