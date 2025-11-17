from server.strategy.fedavg_random_dynff_buffered import FedAvgRandomDynffBuff


class FedProxRandomDynffBuff(FedAvgRandomDynffBuff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
