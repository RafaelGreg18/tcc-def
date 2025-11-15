from server.strategy.fedavg_random_dynff import FedAvgRandomDynff


class FedProxRandomDynff(FedAvgRandomDynff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
