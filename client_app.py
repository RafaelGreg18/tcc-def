import pickle

from flwr.client import ClientApp
from flwr.common import Context

from client.base import BaseClient
from client.critical import CriticalClient
from client.feddyn import BaseFedDynClient
from client.feddyn_critical import FedDynCriticalClient
from client.prox import BaseProxClient
from client.prox_critical import ProxCriticalClient
from utils.simulation.config import set_seed
from utils.simulation.workflow import get_user_dataloader, get_initial_model


def client_fn(context: Context):
    # 1. config
    set_seed(context.run_config["seed"])
    # 2. client id
    cid = context.node_config["partition-id"]
    flwr_cid = str(context.node_id)
    # 3. model
    model = get_initial_model(context)
    # 4. dataloader
    dataloader = get_user_dataloader(context, cid)
    dataset_id = context.run_config["hugginface-id"]
    # 5. client
    is_critical = context.run_config["is-critical"]
    # 6. aggregation type
    agg = context.run_config["aggregation-name"]

    if agg == "fedavg":
        if is_critical:
            return CriticalClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                                  dataset_id=dataset_id).to_client()
        else:
            return BaseClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                              dataset_id=dataset_id).to_client()
    elif agg == "fedprox":
        proximal_mu = context.run_config["proximal-mu"]
        if is_critical:
            return ProxCriticalClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                                      dataset_id=dataset_id, proximal_mu=proximal_mu).to_client()
        else:

            return BaseProxClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                                  dataset_id=dataset_id, proximal_mu=proximal_mu).to_client()
    elif agg == "feddyn":
        with open(f"prev_grads/client_{int(cid)}", "rb") as prev_grads_file:
            prev_grads = pickle.load(prev_grads_file)
        alpha = context.run_config["alpha-coef"]

        if is_critical:
            return FedDynCriticalClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                                        dataset_id=dataset_id, prev_grads=prev_grads, alpha=alpha).to_client()
        else:
            return BaseFedDynClient(cid=cid, flwr_cid=flwr_cid, model=model, dataloader=dataloader,
                                    dataset_id=dataset_id, prev_grads=prev_grads, alpha=alpha).to_client()


app = ClientApp(client_fn)
