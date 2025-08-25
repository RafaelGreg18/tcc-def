import sys
sys.path.append('../')

import torch
import numpy as np

from utils.dataset.fileloader import DataLoaderHelper
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

    selected_by_round = np.random.randint(0, 100, size=(20, 10))
    for row in range(len(selected_by_round)):
        selected_clients = selected_by_round[row]
        for id in selected_clients:
            user_dataset_path = "../dataset/" + f"train_partition_{id.item()}.pt"
            dataset_id = cfg["hugginface-id"]
            dataloader = DataLoaderHelper.load_dataloader_samples(user_dataset_path, g, dataset_id, shuffle=True)
            train(model, dataloader, epochs, criterion, optimizer, device)

    test_path = "../dataset/" + cfg["testset-name"]
    dataset_id = cfg["hugginface-id"]
    testloader = DataLoaderHelper.load_dataloader_samples(test_path, g, dataset_id, shuffle=False)
    avg_loss, avg_acc, stat_util = test(model, testloader, device)

    print(avg_loss, avg_acc, stat_util)

if __name__ == '__main__':
    main()
