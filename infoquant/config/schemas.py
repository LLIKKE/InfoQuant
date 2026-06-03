from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ModelConfig:
    input_model: str
    model_max_length: int = 2048
    num_hidden_layers: int = -1
    bf16: bool = False


@dataclass
class QuantConfig:
    w_bits: int = 16
    a_bits: int = 16
    k_bits: int = 16
    v_bits: int = 16
    w_groupsize: int = -1
    a_groupsize: int = -1
    k_groupsize: int = -1
    v_groupsize: int = -1


@dataclass
class RotationConfig:
    rotate: bool = True
    rotated_matrix_path: Optional[str] = None
    block_diag: int = 2
    aug_token: int = 30
    aug_start: int = 2
    epochs: int = 15
    batch_size: int = 4
    temperature: int = 2
    learning_rates: Sequence[int] = (2, 2, 2)


@dataclass
class LacConfig:
    enabled: bool = False
    cali_epochs: int = 5
    cali_bsz: int = 4
    cali_lr: float = 0.02
    clip_parameter: Optional[str] = None


@dataclass
class EvalConfig:
    zero_shot: bool = True
    tasks: Sequence[str] = ()


@dataclass
class PathConfig:
    log_dir: str = "./"
    exp_name: str = "rotated_matrix"
    cache: Optional[str] = None
    exp_dir: Optional[str] = None

