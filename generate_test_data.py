import torch
import json

from pynanoflare.modules import PlainSequential, CausalDilatedConv1d, ResidualBlock, TCNBlock, MicroTCNBlock, FiLM, ConvClipper
from pynanoflare.rnn import ResGRU, ResLSTM
from pynanoflare.convwaveshaper import ConvWaveshaper
from pynanoflare.tcn import TCN, MicroTCN
from pynanoflare.wavenet import WaveNet

layers = {
    # Layers
    'causaldilatedconv1d': CausalDilatedConv1d(7, 11, 3, 2),
    'convclipper': ConvClipper( 1, 1, 12, 4 ),
    'microtcnblock': MicroTCNBlock( 7, 11, 3, 2),
    'plainsequential': PlainSequential( 7, 11, 8, 3),
    'residualblock': ResidualBlock( 7, 3, 2, True),
    'tcnblock': TCNBlock( 7, 11, 3, 2),
    'film': FiLM(7, 3)
}

mu, sigma = 0.1, 0.9

models = {
    'convwaveshaper': ConvWaveshaper( 4, 6, 12, mu, sigma ),
    'microtcn': MicroTCN(1, 8, 1, 3, 8, 16, 3, mu, sigma),
    'resgru': ResGRU(1, 64, 1, 8, 3, mu, sigma),
    'reslstm': ResLSTM(1, 64, 1, 8, 3, mu, sigma),
    'tcn': TCN(1, 7, 1, 4, 8, 8, 3, mu, sigma),
    'wavenet': WaveNet(1, 7, 1, 4, [1, 2, 4, 8, 16, 32], 1, True, 8, 3, mu, sigma)
}

if __name__ == "__main__":

    for layer_name, layer in layers.items():
        print(layer_name)
        doc = layer.generate_doc()
        with open(f'tests/data/{layer_name}.json', 'w') as file:
            json.dump(doc, file)
        script_module = torch.jit.script(layer)
        script_module.save(f'tests/data/{layer_name}.torchscript')

    for model_name, model in models.items():
        print(model_name)
        doc = model.generate_doc()
        with open(f'tests/data/{model_name}.json', 'w') as file:
            json.dump(doc, file)
        script_module = torch.jit.script(model)
        script_module.save(f'tests/data/{model_name}.torchscript')