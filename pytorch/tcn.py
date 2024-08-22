import torch
import torch.nn as nn
from audiomodel import AudioModel, TCNBlock

class TCN(AudioModel):
    def __init__(self, input_size, output_size, kernel_size, stack_size, norm_mean = 0.0, norm_std = 1.0):
        super().__init__(norm_mean, norm_std)
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stack_size = stack_size
        self.blockStack = nn.ModuleList([TCNBlock(1 if i == 0 else 2**stack_size, 2**stack_size, kernel_size, 2**i) for i in range(stack_size+1)])
        self.linear = nn.Linear(2**stack_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blockStack:
            x = block(x)
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