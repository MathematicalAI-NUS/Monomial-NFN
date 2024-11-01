from typing import Type, Optional, Union
from torch import nn
from nfn.layers import HNPLinear, NPLinear, TupleOp, Pointwise, SimpleLayerNorm, LearnedScale, HNPSLinear, HNPS_SirenLinear
from nfn.layers import GaussianFourierFeatureTransform, IOSinusoidalEncoding, FlattenWeights, UnflattenWeights
import torch

MODE2LAYER = {
    "PT": Pointwise,
    "NP": NPLinear,
    "NP-PosEmb": lambda *args, **kwargs: NPLinear(*args, io_embed=True, **kwargs),
    "HNP": HNPLinear,
    "HNPS": HNPSLinear
}

class MLPTransferNet(nn.Module):
    def __init__(
        self,
        network_spec,
        h_size=1000,
        out_scale=0.01,
        hidden_chan= 128,
        hidden_layers= 3,
        mode= "HNP",
        init_type= "pytorch_default"

    ):
        super().__init__()
        num_params = network_spec.get_num_params()
        self.hnet = nn.Sequential(
            FlattenWeights(network_spec),
            nn.Flatten(start_dim=-2),
            nn.Linear(num_params, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, num_params),
        )
        self.unflatten = UnflattenWeights(network_spec)
        self.scale = LearnedScale(network_spec, out_scale)

    def forward(self, params):
        return self.scale(self.unflatten(self.hnet(params).unsqueeze(-1)))


InpEncTypes = Optional[Union[Type[GaussianFourierFeatureTransform], Type[Pointwise]]]
class TransferNet(nn.Module):
    def __init__(
        self,
        network_spec,
        hidden_chan,
        hidden_layers,
        inp_enc_cls: InpEncTypes=None,
        pos_enc_cls: Optional[Type[IOSinusoidalEncoding]]=None,
        mode="full",
        lnorm=False,
        out_scale=0.01,
        dropout=0,
        special=False,
        init_type="pytorch_default",
        hidden_channels=16
    ):
        super().__init__()
        layers = []
        last_channels=1
        in_channels = 1
        if special:
            layers.append(HNPS_SirenLinear(network_spec, in_channels=1, out_channels=hidden_channels,init_type=init_type))
            layers.append(HNPS_SirenLinear(network_spec, in_channels=hidden_channels, out_channels=last_channels,init_type=init_type))
            in_channels = last_channels
        
        if inp_enc_cls is not None:
            inp_enc = inp_enc_cls(network_spec, in_channels)
            layers.append(inp_enc)
            in_channels = inp_enc.out_channels
        if pos_enc_cls is not None:
            pos_enc = pos_enc_cls(network_spec)
            layers.append(pos_enc)
            in_channels = pos_enc.num_out_chan(in_channels)
        layer_cls = MODE2LAYER[mode]
        
        layers.append(layer_cls(network_spec, in_channels=in_channels, out_channels=hidden_chan))
        if lnorm:
            layers.append(SimpleLayerNorm(network_spec, hidden_chan))
        layers.append(TupleOp(nn.ReLU()))   
        if dropout > 0:
            layers.append(TupleOp(nn.Dropout(dropout)))
        
        for _ in range(hidden_layers - 1):
            layers.append(layer_cls(network_spec, in_channels=hidden_chan, out_channels=hidden_chan))
            if lnorm:
                layers.append(SimpleLayerNorm(network_spec, hidden_chan))
            layers.append(TupleOp(nn.ReLU())) 
        layers.append(layer_cls(network_spec, in_channels=hidden_chan, out_channels=1))
        layers.append(LearnedScale(network_spec, out_scale))
        self.hnet = nn.Sequential(*layers)
        print("test")

    def forward(self, params):
        x = params
        for layer in self.hnet:
            x = layer(x)
        return x