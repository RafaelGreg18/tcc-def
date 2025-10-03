from server.strategy.fedavg_random_aff import FedAvgRandomAFF


class FedProxRandomAFF(FedAvgRandomAFF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
