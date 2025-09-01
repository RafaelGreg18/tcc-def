import ast
import os
import random

try:
    import tomllib as toml
except:
    import tomli as toml

import numpy as np
import torch


class ConfigRepository:
    def __init__(self, path):
        with open(path, "rb") as f:
            self._config = toml.load(f)

    def get(self, *keys, default=None):
        value = self._config
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return default
        return value

    def get_app_config(self):
        return self.get("tool", "flwr", "app", "config", default={})

    @classmethod
    def preprocess_app_config(cls, cfg):

        # Defaults
        # global
        cfg.setdefault("seed", 1)
        cfg.setdefault("root-model-dir", "./model/")
        cfg.setdefault("root-profile-dir", "./profiles/")
        cfg.setdefault("root-outputs-dir", "./outputs/")
        cfg.setdefault("devices-profile-path", "./utils/profiles/Mobilenet_v2.json")
        cfg.setdefault("net-speed-path", "./utils/profiles/bandwidth.json")
        cfg.setdefault("carbon-data-path", "./utils/profiles/carbon.json")

        # dataset
        cfg.setdefault("hugginface-id", "uoft-cs/cifar10")
        cfg.setdefault("dir-alpha", 0.3)
        cfg.setdefault("batch-size", 8)

        # model
        cfg.setdefault("model-name", "simplecnn")
        cfg.setdefault("input-shape", "(3, 32, 32)")
        cfg.setdefault("num-classes", 10)
        cfg.setdefault("epochs", 1)
        cfg.setdefault("learning-rate", 1e-3)

        # device profile
        cfg.setdefault("prefer-time", "UNIFORM")
        cfg.setdefault("prefer-battery", "EQUAL")
        cfg.setdefault("prefer-carbon", "UNIFORM")
        cfg.setdefault("battery-profile-low", 10)
        cfg.setdefault("battery-profile-medium", 20)
        cfg.setdefault("battery-profile-high", 30)
        cfg.setdefault("use-battery", False)
        cfg.setdefault("carbon-region", "United States")
        cfg.setdefault("net-scenario", "US")
        # strategy
        cfg.setdefault("participants-name", "constant")
        cfg.setdefault("selection-name", "random")
        cfg.setdefault("aggregation-name", "fedavg")
        cfg.setdefault("num-clients", 100)
        cfg.setdefault("num-rounds", 2)
        cfg.setdefault("num-participants", 10)
        cfg.setdefault("num-evaluators", 0)

        #two-phase
        cfg.setdefault("num-participants-bcp", 10)
        cfg.setdefault("num-participants-acp", 10)

        # client
        cfg.setdefault("battery-threshold", 0.1)

        # Processing
        # global
        cfg["seed"] = int(cfg["seed"])
        os.makedirs(cfg["root-model-dir"], exist_ok=True)
        os.makedirs(cfg["root-profile-dir"], exist_ok=True)
        os.makedirs(cfg["root-outputs-dir"], exist_ok=True)

        # dataset
        cfg["dir-alpha"] = float(cfg["dir-alpha"])
        cfg["batch-size"] = int(cfg["batch-size"])

        # model
        cfg["input-shape"] = ast.literal_eval(cfg["input-shape"])
        cfg["num-classes"] = int(cfg["num-classes"])
        cfg["epochs"] = int(cfg["epochs"])
        cfg["learning-rate"] = float(cfg["learning-rate"])

        # device profile
        cfg["use-battery"] = bool(cfg["use-battery"])
        cfg["battery-profile-low"] = float(cfg["battery-profile-low"])
        cfg["battery-profile-medium"] = float(cfg["battery-profile-medium"])
        cfg["battery-profile-high"] = float(cfg["battery-profile-high"])

        # strategy
        cfg["num-clients"] = int(cfg["num-clients"])
        cfg["num-rounds"] = int(cfg["num-rounds"])
        cfg["num-participants"] = int(cfg["num-participants"])
        cfg["num-evaluators"] = int(cfg["num-evaluators"])

        #two-phase
        cfg["num-participants-bpc"] = int(cfg["num-participants-bcp"])
        cfg["num-participants-apc"] = int(cfg["num-participants-acp"])

        # client
        cfg["battery-threshold"] = float(cfg["battery-threshold"])

        return cfg

    @classmethod
    def validate_app_config(cls, cfg):
        errors = []

        if cfg["num-clients"] < 2:
            errors.append("num-clients >= 2")
        if cfg["dir-alpha"] <= 0:
            errors.append("dir-alpha > 0")
        if cfg["num-classes"] < 2:
            errors.append("num-classes > 1")
        if cfg["prefer-time"] not in ["SLOW", "EQUAL", "FAST"]:
            errors.append("Device inference time distribution config (prefer-time) must be: SLOW, EQUAL or FAST")
        if cfg["prefer-battery"] not in ["LOW", "MEDIUM", "HIGH", "EQUAL"]:
            errors.append(
                "Device battery distribution config (prefer-battery) must be: LOW, MEDIUM, HIGH or EQUAL")
        if cfg["prefer-carbon"] not in ["LOW", "HIGH", "UNIFORM"]:
            errors.append(
                "Device carbon intensity distribution config (prefer-carbon) must be: LOW, HIGH or UNIFORM")
        if cfg["battery-profile-low"] <= 0 or cfg["battery-profile-medium"] <= 0 or cfg["battery-profile-high"] <= 0:
            errors.append("Some battery profile is <= 0")
        if errors:
            raise ValueError("Config errors:\n" + "\n".join(errors))


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)