from flwr.common import FitIns
from flwr.server.client_proxy import ClientProxy

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.strategy.dynff import ema_online, window_degree, first_stable_idx, Plans, schedule_additions


class FedAvgRandomDynff(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Limits
        self.initial_num_participants = self.num_participants
        self.max_participants = int(self.num_clients * self.context.run_config["max-participants-fraction"])
        self.min_participants = int(
            self.initial_num_participants * self.context.run_config["min-participants-fraction"])
        start_exp_round = int(self.context.run_config["start-exp-round"])  # Force EXP starting in this round
        self.smoothing_alpha = float(self.context.run_config["smoothing-alpha"])
        self.scheduling_gamma = float(self.context.run_config["scheduling-gamma"])
        self.bud_percentual = float(self.context.run_config["bud-percentual"])
        self.W = int(self.context.run_config["W"])
        self.K = int(self.context.run_config["K"])
        self.delta = float(self.context.run_config["delta"])
        self.method = self.context.run_config["method"]
        self.start_after = int(self.context.run_config["start-after"])
        self.c_rates = []  # how to change currente participants
        self.acks = int(self.context.run_config["acks"])  # change participantion only after #acks
        self.num_rounds = int(self.context.run_config["num-rounds"]) + 1
        self.additions = None
        self.num_selected = []

        self.state = {
            "plan": Plans.ECO,
            "stable": False,
            "stable_round": -1,
            "start_exp_round": start_exp_round,
            "e_bud": 0
        }

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_participants, self.num_participants

    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        if 2 <= server_round <= 3:
            min_num_clients = sample_size = self.initial_num_participants
            self.num_selected.append(sample_size)
        elif server_round > 3:
            if server_round % self.acks == 0 and not self.state["stable"]:
                self.update_dynff(server_round)
            elif self.state["stable"]:
                self.update_dynff(server_round)

            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            self.num_selected.append(sample_size)
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def update_dynff(self, server_round):
        stability_reached = self.is_stable(server_round)

        if stability_reached:
            idx = server_round - self.state["stable_round"]
            self.num_participants = self.initial_num_participants + self.additions[idx]

    def is_stable(self, server_round):

        if self.state["stable"]:
            return True
        elif 0 < self.state["start_exp_round"] + 1 <= server_round:
            # Planning
            self.state["stable"] = True
            self.state["stable_round"] = server_round
            self.state["plan"] = Plans.EXP
            avg_j_consumption = sum(self.all_selected_clients_consumption) / len(
                self.all_selected_clients_consumption)
            num_eco = ((server_round - 1) * self.initial_num_participants) - sum(self.num_selected)

            if num_eco > 0:
                self.additions = schedule_additions(avg_j_consumption * num_eco * self.bud_percentual,
                                                    avg_j_consumption,
                                                    self.num_rounds + 1, server_round, gamma=self.scheduling_gamma)
                size_diff = ((self.num_rounds + 1 - server_round) - len(self.additions) + 10)
                if size_diff > 0:
                    to_add = self.additions[-1]
                    for i in range(size_diff):
                        self.additions.append(to_add)
            else:
                self.additions = [0] * (self.num_rounds + 1 - server_round)

            return True
        else:
            accuracies = []

            for round in range(server_round):
                acc = self.performance_metrics_to_save[round]["cen_accuracy"]
                accuracies.append(acc)

            # Performance curve
            smoothed_accs = ema_online(accuracies, self.smoothing_alpha)
            c_rate = window_degree(smoothed_accs, self.W, self.K, self.method)

            # Stability
            resp = first_stable_idx(smoothed_accs, W=self.W, K=self.K, delta=self.delta, metodo=self.method,
                                    start_after=self.start_after)
            # Planning
            avg_j_consumption = sum(self.all_selected_clients_consumption) / len(self.all_selected_clients_consumption)

            if resp == None and c_rate >= 0:
                self.state["plan"] = Plans.ECO
                decrease_margin = self.num_participants - self.min_participants
                to_decrease = max(decrease_margin * c_rate, 1)
                self.num_participants -= to_decrease
                self.num_participants = max(self.min_participants, self.num_participants)
                self.state["stable"] = False
                add_bud = (self.initial_num_participants - self.num_participants) * avg_j_consumption
                self.state["e_bud"] += add_bud

                return False

            elif resp == None and c_rate < 0:
                if self.num_participants < self.initial_num_participants:
                    increase_margin = self.initial_num_participants - self.num_participants
                else:
                    increase_margin = self.max_participants - self.num_participants

                to_increase = max(increase_margin * abs(c_rate), 1)
                self.num_participants += to_increase
                self.num_participants = min(self.max_participants, self.num_participants)
                self.state["plan"] = Plans.REC
                add_bud = (self.initial_num_participants - self.num_participants) * avg_j_consumption
                self.state["e_bud"] += add_bud

                return False

            elif resp != None and self.state["start_exp_round"] + 1 <= 0:
                self.state["stable"] = True
                self.state["stable_round"] = resp
                self.state["plan"] = Plans.EXP

                avg_j_consumption = sum(self.all_selected_clients_consumption) / len(
                    self.all_selected_clients_consumption)
                num_eco = ((server_round - 1) * self.initial_num_participants) - sum(self.num_selected)

                if num_eco > 0:
                    self.additions = schedule_additions(avg_j_consumption * num_eco * self.bud_percentual,
                                                        avg_j_consumption,
                                                        self.num_rounds + 1, server_round, gamma=self.scheduling_gamma)
                    size_diff = ((self.num_rounds + 1 - server_round) - len(self.additions) + 10)
                    if  size_diff > 0:
                        to_add = self.additions[-1]
                        for i in range(size_diff):
                            self.additions.append(to_add)
                else:
                    self.additions = [0] * (self.num_rounds + 1 - server_round)

                return True
