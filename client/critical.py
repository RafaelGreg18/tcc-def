import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from utils.model.manipulation import set_weights, get_weights, test, train_critical


class CriticalClient(NumPyClient):
    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id, **kwargs):
        super().__init__(**kwargs)
        self.cid = cid
        self.model = model
        self.dataloader = dataloader
        self.flwr_cid = flwr_cid
        self.dataset_id = dataset_id

    def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if int(config["server_round"]) == 1:
            set_weights(self.model, parameters)
            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": 0, "acc": 0, "stat_util": 0,
                                                                           "gn": 0}  # fgn
        else:
            # update model weights
            set_weights(self.model, parameters)
            # define train config
            epochs = int(config["epochs"])
            learning_rate = float(config["learning_rate"])
            weight_decay = float(config["weight_decay"])
            momentum = float(config["momentum"])

            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                        weight_decay=weight_decay)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            avg_loss, avg_acc, stat_util, gn = train_critical(self.model, self.dataloader, epochs, criterion,
                                                              optimizer, device, self.dataset_id)

            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": avg_loss, "acc": avg_acc,
                                                                           "stat_util": stat_util, "gn": gn}  # fgn

    def evaluate(self, parameters, config):
        if int(config["server_round"]) > 1:
            set_weights(self.model, parameters)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            avg_loss, avg_acc, stat_util = test(self.model, self.dataloader, device, self.dataset_id)

            return avg_loss, len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                            "loss": avg_loss, "acc": avg_acc, "stat_util": stat_util}
        else:

            return 0.0, len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                       "loss": 0, "acc": 0, "stat_util": 0}
