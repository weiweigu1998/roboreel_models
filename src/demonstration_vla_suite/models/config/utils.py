import os
from omegaconf import DictConfig


def get_demonstration_config(demonstration_yaml_or_cfg: str | DictConfig | None):
    """Load a demonstration YAML from ``models/config/`` or pass through a ``DictConfig``."""
    if demonstration_yaml_or_cfg in ["None", "none"]:
        return None
    if isinstance(demonstration_yaml_or_cfg, str):
        import omegaconf
        config_dir = os.path.join(os.path.dirname(__file__))
        loaded = omegaconf.OmegaConf.load(
            os.path.join(config_dir, demonstration_yaml_or_cfg),
        )
        return loaded
    elif isinstance(demonstration_yaml_or_cfg, DictConfig):
        return demonstration_yaml_or_cfg
    elif demonstration_yaml_or_cfg is None:
        return None
    else:
        raise ValueError(f"Invalid demonstration config: {demonstration_yaml_or_cfg}")
