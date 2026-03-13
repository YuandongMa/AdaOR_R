from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderBioVisionCfg, EncoderBioVision
from .encoder_noposplat_multi import EncoderBioVisionMulti
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderBioVision, None),
    "noposplat_multi": (EncoderBioVisionMulti, None),
}

EncoderCfg = EncoderBioVisionCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
