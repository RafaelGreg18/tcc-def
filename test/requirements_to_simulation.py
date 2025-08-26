import sys
sys.path.append('../')
import torch
import numpy as np

from utils.dataset.partition import DatasetFactory
from utils.model.manipulation import ModelPersistence, train, test
from utils.simulation.config import ConfigRepository, set_seed


def main():
    config_file = "../pyproject.toml"

    config_repo = ConfigRepository(config_file)
    cfg = config_repo.get_app_config()
    cfg = config_repo.preprocess_app_config(cfg)
    config_repo.validate_app_config(cfg)

    seed = cfg["seed"]
    set_seed(seed)

    root_model_dir = "../model/"
    model_name = cfg["model-name"]
    input_shape = cfg["input-shape"]
    num_classes = cfg["num-classes"]
    epochs = cfg["epochs"]
    learning_rate = cfg["learning-rate"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = root_model_dir + model_name + '.pth'
    model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    g = torch.Generator()
    g.manual_seed(seed)

    dataset_id = cfg["hugginface-id"]
    num_partitions = cfg["num-clients"]
    alpha = cfg["dir-alpha"]
    batch_size = cfg["batch-size"]


    selected_by_round = np.random.randint(0, 903, size=(20, 42))
    for row in range(len(selected_by_round)):
        selected_clients = selected_by_round[row]
        cliente = 1
        for id in selected_clients:
            dataloader = DatasetFactory.get_partition(dataset_id, id, num_partitions, alpha, batch_size, seed)
            train(model, dataloader, epochs, criterion, optimizer, device, dataset_id)
            print(f"Rodada {row + 1} Cliente {cliente}")
            cliente += 1

    testloader = DatasetFactory.get_test_dataset(dataset_id, batch_size, num_partitions, alpha, seed)
    avg_loss, avg_acc, stat_util = test(model, testloader, device, dataset_id)

    print(avg_loss, avg_acc, stat_util)


if __name__ == '__main__':
    main()
