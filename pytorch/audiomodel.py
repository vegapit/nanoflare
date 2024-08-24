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

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation # Add required padding on each side
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        y = self.conv1d(x)
        return y[ ..., :-self.padding] # discard right padding to preserve signal length and ensure causality
    
class ResidualBlock(nn.Module):
    def __init__(self, num_channels, kernel_size, dilation, gated):
        super().__init__()
        self.num_channels = num_channels
        self.gated = gated
        self.inputConv = CausalDilatedConv1d(num_channels, 2 * num_channels if gated else num_channels, kernel_size, dilation=dilation)
        self.outputConv = nn.Conv1d(num_channels, num_channels, 1)
        self.f = nn.Tanh()
        self.g = nn.Sigmoid() # Gate activation function
        
    def forward(self, x):
        #print(f"ResBlock: {x.shape}")
        if self.gated:
            ys = torch.split( self.inputConv(x), self.num_channels, dim=0) # Separate Filter and Gate
            y = self.f( ys[0] ) * self.g( ys[1] )
        else:
            y = self.f( self.inputConv(x) )
        y = self.outputConv( y )
        return y + x, y

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.conv1 = CausalDilatedConv1d( in_channels, out_channels, kernel_size, dilation=dilation )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalDilatedConv1d( out_channels, out_channels, kernel_size, dilation=1 )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.f = nn.ELU()

    def forward(self, x: torch.Tensor):
        y = self.bn1(self.f(self.conv1(x)))
        y = self.bn2(self.f(self.conv2(y)))
        if(self.in_channels == self.out_channels):
            return x + y
        else:
            return self.conv(x) + y
    
def error_to_signal(y, y_pred):
    """
    Error to signal ratio with pre-emphasis filter:
    https://www.mdpi.com/2076-3417/10/3/766/htm
    """
    y, y_pred = pre_emphasis_filter(y), pre_emphasis_filter(y_pred)
    return (y - y_pred).pow(2).sum(dim=2) / y.pow(2).sum(dim=2)

def pre_emphasis_filter(x, coeff=0.95): # for n >= 1, y[n] = x[n] - 0.95 * x[n-1] or y[0] = x[0]
    return torch.cat((x[:, :, 0:1], x[:, :, 1:] - coeff * x[:, :, :-1]), dim=2)