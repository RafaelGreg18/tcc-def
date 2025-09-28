from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedProxRandomConstant(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
