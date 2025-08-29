def get_selected_cids_and_local_training_data_size(results):
    selected_cids = {}

    for result in results:
        ds_size = result[1].num_examples
        cid = result[1].metrics["cid"]
        selected_cids[cid] = ds_size

    return selected_cids


def get_training_time_per_cid(client_profile, client_dataset_size, model_size, epochs):
    # Proc: using times from https://ai-benchmark.com/archive/ranking_detailed_5_1_2.html
    # Comm: USA - https://www.opensignal.com/reports/2025/01/usa/mobile-network-experience
    # Comm: BR https://www.opensignal.com/reports/2025/01/brazil/mobile-network-experience

    cids_time = {cid: {"proc": 0, "comm": 0, "total": 0} for cid in client_dataset_size.keys()}

    max_comm_round_time = 0
    for cid in cids_time:
        ds_size = client_dataset_size[cid]
        proc = client_profile[cid]["training_ms"] * ds_size * epochs
        comm = model_size / client_profile[cid]["up_speed"] + model_size / client_profile[cid]["down_speed"]
        cids_time[cid]["proc"] = proc
        cids_time[cid]["comm"] = comm * 1000  # s -> ms
        cids_time[cid]["total"] = proc + comm * 1000  # s -> ms
        max_comm_round_time = max(max_comm_round_time, proc + comm * 1000)  # s -> ms

    return cids_time, max_comm_round_time


def get_selected_cid_training_energy(client_profile, selected_cids_time, client_dataset_size, model_size, epochs,
                                     max_comm_round_time, use_battery):
    # Proc joules: https://burnout-benchmark.com/ranking_power_efficiency.html 1W = 1J/s
    # Comm joules: Understanding Operational 5G: A First Measurement Study on Its Coverage, Performance and Energy Consumption

    cid_joule_consumption = {cid: 0 for cid in selected_cids_time.keys()}

    for cid in selected_cids_time.keys():
        ds_size = client_dataset_size[cid]
        proc_joules = client_profile[cid]["training_mJ"] * ds_size * epochs
        comm_joules = model_size * client_profile[cid]["comm_joules"] * 2  # up+down
        idle_time = max_comm_round_time - selected_cids_time[cid]["total"]
        idle_joules = idle_time * client_profile[cid]["idle_mJ"]
        total = proc_joules + comm_joules + idle_joules
        if use_battery:
            cid_joule_consumption[cid] = min(total, client_profile[cid]["current_battery_mJ"])
        else:
            cid_joule_consumption[cid] = total

    return cid_joule_consumption


def get_cid_training_carbon_footprint(client_profile, selected_cids_joules_consumption):
    cid_carbon_footprint = {cid: 0 for cid in selected_cids_joules_consumption.keys()}

    for cid in selected_cids_joules_consumption.keys():
        # Converta X (mJ) para kWh usando kWh=3,6MJ=3,6×10^6J
        # Pegada (gCO₂e) = energia (kWh) × intensidade carbônica Z (gCO₂e/kWh)
        # — esta é a fórmula padrão “atividade × fator de emissão”.
        carbon_footprint = client_profile[cid]["device_carbon_intensity"] * (
                    selected_cids_joules_consumption[cid] / (3.6 * (10 ** 9)))  # J -> kWh
        cid_carbon_footprint[cid] = carbon_footprint

    return cid_carbon_footprint


def get_unselected_cid_consumption(client_profile, unselected_cids, elapsed_time, use_battery):
    cid_joule_consumption = {cid: 0 for cid in unselected_cids}

    for cid in cid_joule_consumption.keys():
        idle_joules = elapsed_time * client_profile[cid][
            "idle_mJ"]  # converter perfil de bateria para mJ e atualizar códigos anteriores.

        if use_battery:
            cid_joule_consumption[cid] = min(idle_joules, client_profile[cid]["current_battery_mJ"])
        else:
            cid_joule_consumption[cid] = idle_joules

    return cid_joule_consumption
