from server.strategy.fedavg_random_hetaaff import FedAvgRandomHETAAFF


class FedProxRandomHETAAFF(FedAvgRandomHETAAFF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
