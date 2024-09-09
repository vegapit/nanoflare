import torch
import json

from audiomodel import PlainSequential, CausalDilatedConv1d, ResidualBlock, TCNBlock, MicroTCNBlock 
from rnn import ResGRU, ResLSTM
from tcn import TCN, MicroTCN
from wavenet import WaveNet

layers = {
    # Layers
    'causaldilatedconv1d': CausalDilatedConv1d(7, 11, 3, 2),
    'tcnblock': MicroTCNBlock( 7, 11, 3, 2),
    'plainsequential': PlainSequential( 7, 11, 8, 3),
    'residualblock': ResidualBlock( 7, 3, 2, True),
    'tcnblock': TCNBlock( 7, 11, 3, 2),
}

mu, sigma = 0.1, 0.9

models = {
    'microtcn': MicroTCN(1, 8, 1, 3, 8, 16, 3, mu, sigma),
    'resgru': ResGRU(1, 64, 1, 8, 3, mu, sigma),
    'reslstm': ResLSTM(1, 64, 1, 8, 3, mu, sigma),
    'tcn': TCN(1, 8, 1, 3, 8, 8, 3, mu, sigma),
    'wavenet': WaveNet(1, 8, 1, 3, [1, 2, 4, 8, 16, 32, 64, 128], 1, True, 8, 3, mu, sigma)
}

if __name__ == "__main__":

    for layer_name, layer in layers.items():
        print(layer_name)
        doc = layer.generate_doc()
        with open(f'test_data/{layer_name}.json', 'w') as file:
            json.dump(doc, file)
        script_module = torch.jit.script(layer)
        script_module.save(f'test_data/{layer_name}.torchscript')

    for model_name, model in models.items():
        print(model_name)
        doc = model.generate_doc()
        with open(f'test_data/{model_name}.json', 'w') as file:
            json.dump(doc, file)
        script_module = torch.jit.script(model)
        script_module.save(f'test_data/{model_name}.torchscript')