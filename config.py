from typing import Union, Optional, Tuple, Sequence, List

from tensorfn.config import (
    get_models,
    get_model,
    MainConfig,
    Config,
    Optimizer,
    Scheduler,
    DataLoader,
    checker,
    Checker,
    TypedConfig,
    Instance,
)
from pydantic import StrictStr, StrictInt, StrictBool


class Training(Config):
    size: StrictInt
    iter: StrictInt = 800000
    batch: StrictInt = 16
    n_sample: StrictInt = 32
    r1: float = 10
    d_reg_every: StrictInt = 16
    lr_g: float = 2e-3
    lr_d: float = 2e-3
    augment: StrictBool = False
    augment_p: float = 0
    ada_target: float = 0.6
    ada_length: StrictInt = 500 * 1000
    ada_every: StrictInt = 256
    start_iter: StrictInt = 0


class GANConfig(MainConfig):
    generator: Instance
    discriminator: Instance
    training: Training
    path: StrictStr = None
    wandb: StrictBool = False
    logger: StrictStr = "rich"
