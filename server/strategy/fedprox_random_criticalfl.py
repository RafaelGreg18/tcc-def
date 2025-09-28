from server.strategy.fedavg_random_criticalfl import FedAvgRandomCriticalFL


class FedProxRandomCriticalFL(FedAvgRandomCriticalFL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
