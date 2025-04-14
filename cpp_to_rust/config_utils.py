import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load and return a configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    
    with config_file.open("r") as f:
        config = yaml.safe_load(f)
        config["training"]["learning_rate"] = float(config["training"]["learning_rate"])
        config["training"]["adam_epsilon"] = float(config["training"]["adam_epsilon"])
        return config