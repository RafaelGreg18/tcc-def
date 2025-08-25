import ast
import json
from typing import Dict, Any, List, Tuple, Callable

import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics, Parameters, MetricsAggregationFn
from flwr.server import ServerConfig, Server, SimpleClientManager, ServerAppComponents
from torch.utils.data import DataLoader

from utils.dataset.fileloader import DataLoaderHelper
from utils.model.manipulation import ModelPersistence, get_weights, set_weights, test
from utils.simulation.config import ConfigRepository


def config_preprocess_validation(context: Context):
    cfg = context.run_config
    ConfigRepository.preprocess_app_config(cfg)
    ConfigRepository.validate_app_config(cfg)


def get_initial_parameters(context: Context):
    model_name = context.run_config['model-name']
    input_shape = context.run_config['input-shape']
    num_classes = context.run_config['num-classes']
    root_model_dir = context.run_config["root-model-dir"]
    model_path = root_model_dir + model_name + '.pth'
    loaded_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
    ndarrays = get_weights(loaded_model)
    parameters = ndarrays_to_parameters(ndarrays)

    return parameters


def get_initial_model(context: Context):
    model_name = context.run_config['model-name']
    input_shape = ast.literal_eval(context.run_config['input-shape'])
    num_classes = context.run_config['num-classes']
    root_model_dir = context.run_config["root-model-dir"]
    model_checkpoint = context.run_config["model-checkpoint"]
    model_path = root_model_dir + model_name + f'_{model_checkpoint}.pth'
    loaded_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)

    return loaded_model


def get_model_memory_size_bits(context: Context):
    """
    Computes the model's size in bits.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Model size in bits.
    """
    model_name = context.run_config['model-name']
    input_shape = context.run_config['input-shape']
    num_classes = context.run_config['num-classes']
    root_model_dir = context.run_config["root-model-dir"]
    model_path = root_model_dir + model_name + '.pth'
    model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
    size_in_bits = sum(p.numel() * p.element_size() * 8 for p in model.parameters())

    return size_in_bits


def get_central_testloader(context: Context):
    test_path = context.run_config["root-data-dir"] + context.run_config["testset-name"]
    dataset_id = context.run_config["hugginface-id"]
    seed = context.run_config["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    test_loader = DataLoaderHelper.load_dataloader_samples(test_path, g, dataset_id, shuffle=False)

    return test_loader


def get_user_dataloader(context: Context, cid):
    user_dataset_path = context.run_config["root-data-dir"] + f"train_partition_{cid}.pt"
    dataset_id = context.run_config["hugginface-id"]
    seed = context.run_config["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DataLoaderHelper.load_dataloader_samples(user_dataset_path, g, dataset_id, shuffle=True)

    return dataloader


def get_eval_fn(context: Context, test_loader: DataLoader):
    def evaluate(server_round, parameters_ndarrays, config):
        model_name = context.run_config['model-name']
        input_shape = context.run_config['input-shape']
        num_classes = context.run_config['num-classes']
        root_model_dir = context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
        set_weights(model, parameters_ndarrays)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, acc, _ = test(model, test_loader, num_classes, device)
        return loss, {"cen_accuracy": acc}

    return evaluate


def get_on_fit_config_fn(context: Context):
    epochs = int(context.run_config["epochs"])
    learning_rate = float(context.run_config["learning-rate"])

    def on_fit_config(server_round: int) -> Dict[str, Any]:
        return {"epochs": epochs, "learning_rate": learning_rate}

    return on_fit_config


def get_fit_metrics_aggregation_fn():
    def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    return handle_fit_metrics


def get_strategy(context: Context, initial_parameters: Parameters, fit_metrics_aggregation_fn: MetricsAggregationFn,
                 on_fit_config_fn: Callable, evaluate_fn: Callable):
    selection_name = context.run_config["selection-name"]
    aggregation_name = context.run_config["aggregation-name"]
    fraction_fit = float(context.run_config["fraction-fit"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    profiles = get_profiles(context)

    if aggregation_name == "fedavg":
        if selection_name == "random":
            pass

    return strategy


def get_profiles(context):
    profiles_path = context.run_config[
                        "root-profiles-dir"] + f"profiles_{context.run_config['profile-checkpoint']}.json"
    with open(profiles_path, "r") as file:
        profiles = json.load(file)
    profiles = {int(k): v for k, v in profiles.items()}
    return profiles


def get_server_app_components(context, strategy):
    if context.run_config["is-stat-sys"] and not context.run_config["enable-checkpoints"]:
        num_rounds = context.run_config["num-rounds"] - context.run_config["model-checkpoint"]
    else:
        num_rounds = context.run_config["num-rounds"]

    config = ServerConfig(num_rounds=num_rounds)
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    server.set_max_workers(int(0.1 * int(context.run_config["num-clients"])))
    components = ServerAppComponents(strategy=strategy, config=config, server=server)
    return components