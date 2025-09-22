import ast
import json
from typing import Dict, Any, List, Tuple, Callable

import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics, Parameters, MetricsAggregationFn
from flwr.server import ServerConfig, Server, SimpleClientManager, ServerAppComponents
from torch.utils.data import DataLoader

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from server.strategy.fedavg_random_constant_twophase import FedAvgRandomConstantTwoPhase
from server.strategy.fedavg_random_criticalfl import FedAvgRandomCPEval
from server.strategy.fedavg_random_recombination import FedAvgRandomRecombination
from utils.dataset.partition import DatasetFactory
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
    model_path = root_model_dir + model_name + '.pth'
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
    dataset_id = context.run_config["hugginface-id"]
    batch_size = context.run_config["batch-size"]
    num_partitions = context.run_config["num-clients"]
    dir_alpha = context.run_config["dir-alpha"]
    seed = context.run_config["seed"]

    g = torch.Generator()
    g.manual_seed(seed)

    test_loader, proxy_loader = DatasetFactory.get_test_dataset(dataset_id, batch_size, num_partitions, dir_alpha, seed)

    return test_loader, proxy_loader


def get_user_dataloader(context: Context, cid):
    dataset_id = context.run_config["hugginface-id"]
    num_partitions = context.run_config["num-clients"]
    dir_alpha = context.run_config["dir-alpha"]
    batch_size = context.run_config["batch-size"]
    seed = context.run_config["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    dataloader = DatasetFactory.get_partition(dataset_id, cid, num_partitions, dir_alpha, batch_size, seed)

    return dataloader


def get_eval_fn(context: Context, test_loader: DataLoader):
    def evaluate(server_round, parameters_ndarrays, config):
        dataset_id = context.run_config['hugginface-id']
        model_name = context.run_config['model-name']
        input_shape = context.run_config['input-shape']
        num_classes = context.run_config['num-classes']
        root_model_dir = context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
        set_weights(model, parameters_ndarrays)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        loss, acc, _ = test(model, test_loader, device, dataset_id)
        return loss, {"cen_accuracy": acc}

    return evaluate


def get_on_fit_config_fn(context: Context):
    epochs = int(context.run_config["epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    weight_decay = float(context.run_config["weight-decay"])
    participants_name = context.run_config["participants-name"]
    decay_step = int(context.run_config["decay-step"])
    momentum = float(context.run_config["momentum"])

    def on_fit_config(server_round: int) -> Dict[str, Any]:
        # testing fgn
        if server_round % decay_step == 0:
            mul_factor = server_round // decay_step
            lr = learning_rate
            for _ in range(mul_factor):
                lr *= weight_decay
        else:
            lr = learning_rate

        return {"server_round": server_round, "epochs": epochs, "learning_rate": lr,
                "weight_decay": weight_decay, "participants_name": participants_name,
                "momentum": momentum}

    return on_fit_config


def get_on_eval_config_fn(context: Context):
    def on_eval_config(server_round: int) -> Dict[str, Any]:
        return {"server_round": server_round}

    return on_eval_config


def get_fit_metrics_aggregation_fn(is_critical: bool):
    def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        if is_critical:
            # fgn
            gns = [m["gn"] for _, m in metrics]
            # Aggregate and return custom metric (weighted average)
            return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples),
                    "avg_gn": sum(gns) / len(gns)}
        else:
            return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    return handle_fit_metrics


def get_evaluate_metrics_aggregation_fn():
    def handle_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        # Multiply accuracy of each client by number of examples used
        accuracies = [num_examples * m["acc"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        # Aggregate and return custom metric (weighted average)
        return {"acc": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

    return handle_eval_metrics


def get_strategy(context: Context, initial_parameters: Parameters, fit_metrics_aggregation_fn: MetricsAggregationFn,
                 evaluate_metrics_aggregation_fn: MetricsAggregationFn, on_fit_config_fn: Callable,
                 on_eval_config_fn: Callable, evaluate_fn: Callable, proxy_loader: DataLoader):
    participants_name = context.run_config["participants-name"]
    selection_name = context.run_config["selection-name"]
    aggregation_name = context.run_config["aggregation-name"]
    num_clients = int(context.run_config["num-clients"])
    num_participants = int(context.run_config["num-participants"])
    num_evaluators = int(context.run_config["num-evaluators"])
    profiles = get_profiles(context)

    if aggregation_name == "fedavg":
        if selection_name == "random":
            if participants_name == "constant":
                strategy = FedAvgRandomConstant(repr="FedAvgRandomConstant",
                                                num_clients=num_clients,
                                                profiles=profiles,
                                                num_participants=num_participants,
                                                num_evaluators=num_evaluators,
                                                context=context,
                                                initial_parameters=initial_parameters,
                                                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                                on_fit_config_fn=on_fit_config_fn,
                                                on_eval_config_fn=on_eval_config_fn,
                                                evaluate_fn=evaluate_fn)
            elif participants_name == "twophase":
                strategy = FedAvgRandomConstantTwoPhase(repr="FedAvgRandomConstantTwoPhase",
                                                        num_clients=num_clients,
                                                        profiles=profiles,
                                                        num_participants=num_participants,
                                                        num_evaluators=num_evaluators,
                                                        context=context,
                                                        initial_parameters=initial_parameters,
                                                        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                                        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                                        on_fit_config_fn=on_fit_config_fn,
                                                        on_eval_config_fn=on_eval_config_fn,
                                                        evaluate_fn=evaluate_fn)
            elif participants_name == "criticalfl":
                strategy = FedAvgRandomCPEval(repr="CriticalFL",
                                              num_clients=num_clients,
                                              profiles=profiles,
                                              num_participants=num_participants,
                                              num_evaluators=num_evaluators,
                                              context=context,
                                              initial_parameters=initial_parameters,
                                              fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                              evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                              on_fit_config_fn=on_fit_config_fn,
                                              on_eval_config_fn=on_eval_config_fn,
                                              evaluate_fn=evaluate_fn)
            elif participants_name == "recombination":
                strategy = FedAvgRandomRecombination(repr="FedAvgRandomRecombination",
                                                     num_clients=num_clients,
                                                     profiles=profiles,
                                                     num_participants=num_participants,
                                                     num_evaluators=num_evaluators,
                                                     context=context,
                                                     initial_parameters=initial_parameters,
                                                     fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                                                     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                                                     on_fit_config_fn=on_fit_config_fn,
                                                     on_eval_config_fn=on_eval_config_fn,
                                                     evaluate_fn=evaluate_fn,
                                                     proxy_loader=proxy_loader)

    return strategy


def get_server_app_components(context, strategy):
    num_rounds = context.run_config["num-rounds"] + 1

    config = ServerConfig(num_rounds=num_rounds)
    server = Server(strategy=strategy, client_manager=SimpleClientManager())
    server.set_max_workers(int(0.1 * int(context.run_config["num-clients"])))
    components = ServerAppComponents(strategy=strategy, config=config, server=server)
    return components


def get_profiles(context):
    profiles_path = context.run_config[
                        "root-profiles-dir"] + "profiles.json"
    with open(profiles_path, "r") as file:
        profiles = json.load(file)
    profiles = {int(k): v for k, v in profiles.items()}
    return profiles
