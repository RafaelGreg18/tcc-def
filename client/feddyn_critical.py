import pickle

import torch
from flwr.common import NDArrays, Scalar

from client.feddyn import BaseFedDynClient
from utils.model.manipulation import set_weights, get_weights, train_feddyn_critical


class FedDynCriticalClient(BaseFedDynClient):
    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id, prev_grads, alpha, **kwargs):
        super().__init__(cid, flwr_cid, model, dataloader, dataset_id, prev_grads, alpha, **kwargs)

    def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if int(config["server_round"]) == 1:
            set_weights(self.model, parameters)
            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": 0, "acc": 0, "stat_util": 0, "gn": 0}
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

            avg_loss, avg_acc, stat_util, self.prev_grads, gn = train_feddyn_critical(self.model, self.dataloader,
                                                                                      epochs,
                                                                                      criterion,
                                                                                      optimizer, device,
                                                                                      self.dataset_id,
                                                                                      self.prev_grads,
                                                                                      self.alpha)

            with open(f"prev_grads/client_{self.cid}", "wb") as prev_grads_file:
                pickle.dump(self.prev_grads, prev_grads_file)

            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": avg_loss, "acc": avg_acc,
                                                                           "stat_util": stat_util, "gn": 0}
