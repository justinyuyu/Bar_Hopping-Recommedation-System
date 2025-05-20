import yaml
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

@dataclass
class AdapterConfig:
    # Training parameters
    input_dim: int = 1024
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.0001
    warmup_steps: int = 100
    margin: float = 0.5
    device: str = "cpu"

    num_bars: int = 100
    questions_per_bar: int = 10

    eval_k: int = 20
    model_save_path: str = "adapter_model.pt"
    data_dir: str = "data"

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'AdapterConfig':
        """Load config from a YAML file."""
        yaml_path = Path(yaml_path)
        with yaml_path.open("r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save config to a YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yaml_path.open('w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

default_config = AdapterConfig()

def get_config(config_path: Optional[Path] = None) -> AdapterConfig:
    """Load config from file if available; otherwise return default."""
    if config_path and Path(config_path).exists():
        return AdapterConfig.from_yaml(config_path)
    return default_config