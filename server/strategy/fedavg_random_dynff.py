import json
from enum import Enum
from logging import WARNING
from typing import Optional

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate
from numpy.ma.core import arctan
from scipy.interpolate import UnivariateSpline

from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class Plans(Enum):
    ECO = 0
    REC = 1
    EXP = 2

class FedAvgRandomDynff(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Limits
        self.initial_num_participants = self.num_participants
        self.max_participants = int(self.num_clients * self.context.run_config["max-participants-fraction"])
        self.min_participants = int(
            self.initial_num_participants * self.context.run_config["min-participants-fraction"])
        self.start_exp_round = int(self.context.run_config["start-exp-round"]) # Force EXP starting in this round
