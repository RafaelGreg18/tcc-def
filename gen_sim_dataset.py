import argparse

from utils.dataset.fileloader import DataLoaderHelper
from utils.dataset.partition import DatasetFactory
from utils.simulation.config import ConfigRepository, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="./pyproject.toml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Read config simulation file and validate it
    config_repo = ConfigRepository(args.config_file)
    cfg = config_repo.get_app_config()
    cfg = config_repo.preprocess_app_config(cfg)
    config_repo.validate_app_config(cfg)

    # Using seed
    set_seed(args.seed)

    # dataset info for posterior process

    for partition_id in range(cfg["num-clients"]):
        # Cria o dataloader da partição
        trainloader = DatasetFactory.get_partition(
            dataset_id=cfg["hugginface-id"],
            partition_id=partition_id,
            num_partitions=cfg["num-clients"],
            alpha=cfg["dir-alpha"],
            batch_size=cfg["batch-size"],
            seed=args.seed,
        )

        # Salva a partição de treino
        DataLoaderHelper.save_dataloader_samples(
            trainloader, f"{cfg['root-data-dir']}/train_partition_{partition_id}.pt"
        )

        print(f"Client {partition_id} done.")

    # Cria e salva o dataloader de teste global (não-particionado)
    testloader = DatasetFactory.get_test_dataset(
        dataset_id=cfg["hugginface-id"],
        batch_size=cfg["batch-size"],
        num_partitions=cfg["num-clients"],  # mantém o mesmo num_partitions para consistência
        alpha=cfg["dir-alpha"],
        seed=args.seed,
    )

    DataLoaderHelper.save_dataloader_samples(
        testloader, f"{cfg['root-data-dir']}/test_global.pt"
    )


if __name__ == "__main__":
    main()