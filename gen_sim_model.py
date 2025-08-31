import argparse
from ast import literal_eval

from utils.model.factory import ModelFactory
from utils.model.manipulation import ModelPersistence
from utils.simulation.config import ConfigRepository, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./pyproject.toml")
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--model_name", type=str, default="simplecnn")
    # parser.add_argument("--input_shape", type=str, default="(3,32,32)")
    #     parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    # Read config simulation file and validate it
    config_repo = ConfigRepository(args.config_file)
    cfg = config_repo.get_app_config()
    cfg = config_repo.preprocess_app_config(cfg)
    config_repo.validate_app_config(cfg)

    # Using seed
    set_seed(args.seed)

    # Get valid config
    cfg = config_repo.get_app_config()
    model_name = cfg["model-name"]
    input_shape = cfg["input-shape"]
    num_classes = cfg["num-classes"]

    # Creating a model
    model = ModelFactory.create(model_name=model_name, input_shape=input_shape, num_classes=num_classes)

    # Saving
    root_model_dir = cfg["root-model-dir"]
    saving_path = root_model_dir + model_name + '.pth'
    ModelPersistence.save(model, saving_path)

if __name__ == "__main__":
    main()