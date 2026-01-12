import os
import yaml
from src.core.config import AppConfig

def test_config_loading():
    # Test loading from non-existent file (should use defaults)
    config = AppConfig.load("non_existent.yaml")
    assert config.algorithm == "mcts"
    assert config.training.batch_size == 256

    # Test loading from a temporary yaml
    temp_yaml = "temp_config.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump({"algorithm": "grpo", "training": {"batch_size": 128}}, f)

    config = AppConfig.load(temp_yaml)
    assert config.algorithm == "grpo"
    assert config.training.batch_size == 128

    os.remove(temp_yaml)
    print("Config tests passed!")

if __name__ == "__main__":
    test_config_loading()
